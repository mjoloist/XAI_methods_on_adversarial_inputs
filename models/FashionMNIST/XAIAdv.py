import csv
import os
from textwrap import wrap
import cv2
import foolbox as fb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from foolbox.attacks import PGD, LinfDeepFoolAttack, LinfBasicIterativeAttack, L2CarliniWagnerAttack, SpatialAttack, \
    EADAttack, FGSM
from packaging import version
from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus, XGradCAM, AblationCAM, ScoreCAM, \
    EigenCAM, EigenGradCAM, LayerCAM, FullGrad, DeepFeatureFactorization, KPCA_CAM, FEM, ShapleyCAM, FinerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, show_factorization_on_image
from scipy.stats import pearsonr
from torch import manual_seed
from torchvision import datasets
from torchvision.transforms import ToTensor

from classifier.FashionMNISTModel import FashionMNISTModelV1

# Important: Matplotlib version <3.10.0 needed due to a mistake in grad-cam that returns an error for show_factorization_on_image
if version.parse(matplotlib.__version__) >= version.parse("3.9.0"):
    raise ImportError(
        f"Matplotlib<3.9.0 is required due to grad-cam, but found version == {matplotlib.__version__}"
    )

# Little Workaround to get ROCm to work with certain older amd graphics cards, since you else-wise get a HIP error for unsupported functions
os.putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else torch.device('cpu')
print(device)






def grad_cam_func(targets, selected_cam, model, input_img, classes, label, classifier, device="cpu"):
    """
    This function produces a visualization for a given model and sample using a chosen visualization method
    :param targets: target layers to be used in methods where it is relevant
    :param selected_cam: index of the selected cam method from the array [GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus,
                     XGradCAM, AblationCAM, ScoreCAM, EigenCAM, EigenGradCAM,
                     LayerCAM, FullGrad, KPCA_CAM,
                     FEM, ShapleyCAM, FinerCAM, DeepFeatureFactorization]
    :param model: the pytorch model for the visualization
    :param input_img: image for which the visualization should be generated
    :param classes: array of all classes in the dataset
    :param label: the correct label of input_img
    :param classifier: the global average pooling layer of the model
    :param device: the device where everything should be computed
    :return: the visualization heatmap on the input_img, the name of the method used, the raw visualization
    """
    model.eval()
    methods_array = [GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus,
                     XGradCAM, AblationCAM, ScoreCAM, EigenCAM, EigenGradCAM,
                     LayerCAM, FullGrad, KPCA_CAM,
                     FEM, ShapleyCAM, FinerCAM]
    # DFF needs to be handled separately due to it having different outputs etc.
    if selected_cam == 15:
        return deep_feature_factorization(targets, model, input_img, classes, label, classifier, device)
    else:
        #ShapleCAM only wants a single layer
        if selected_cam == 13:
            last_layer = targets[-1]
            targets = [last_layer]
        method = methods_array[selected_cam](model=model, target_layers=targets)
        # Transforms the image to NHWC
        transformed_image = input_img.permute(1, 2, 0)
        cam = method(input_tensor=input_img.unsqueeze(dim=1))
        cam = cam[0, :]
        #generates heatmap on input image
        visualization = show_cam_on_image(transformed_image.cpu().numpy(), cam, use_rgb=True)
        print(method.__class__.__name__)
        return visualization, method.__class__.__name__, cam


def deep_feature_factorization(targets, model, input_img, classes, label, classifier_layer, device="cpu"):
    """
    A special implementation of grad_cam_func for Deep Feature Factorization, as it needs to be handled differently
    :param targets:
    :param model:
    :param input_img:
    :param classes:
    :param label:
    :param classifier_layer:
    :param device:
    :return:
    """
    model.eval()
    ## some layers need to be moved to the cpu due to an error in the grad-cam library, and therefore everything has to be on the cpu
    classifier_layer = classifier_layer.to("cpu")
    model = model.to("cpu")
    input_img = input_img.to("cpu")
    dff = DeepFeatureFactorization(model=model, target_layer=targets[len(targets) - 1].to("cpu"),
                                   computation_on_concepts=classifier_layer)
    # Transforms the image to NHWC
    transformed_image = input_img.permute(1, 2, 0)
    concepts, batch_explanations, concept_outputs = dff(input_img.unsqueeze(dim=1), len(classes))
    # Generates the two most likely labels for the different features
    concept_outputs = torch.softmax(input=torch.from_numpy(concept_outputs), dim=-1).numpy()
    concept_label = create_dff_labels(concept_outputs, 2, class_names)
    # scales the image and explanations up in order for the legend to be legible
    transformed_image = torch.tensor(
        cv2.resize(transformed_image.numpy(), (224, 224), interpolation=cv2.INTER_LINEAR)).unsqueeze(dim=2)
    resized_explanations = []
    for exp in batch_explanations[0]:
        resized_explanations.append(cv2.resize(exp, (224, 224), interpolation=cv2.INTER_LINEAR))
    resized_explanations = np.array(resized_explanations)
    visualization = show_factorization_on_image(transformed_image.cpu().numpy(), resized_explanations, image_weight=0.3,
                                                concept_labels=concept_label)
    print(dff.__class__.__name__)
    # moves everything back to device
    targets[len(targets) - 1] = targets[len(targets) - 1].to(device)
    classifier_layer = classifier_layer.to(device)
    model = model.to(device)
    input_img = input_img.to(device)
    return visualization, dff.__class__.__name__


def create_dff_labels(scores, k, labels):
    """
    Largely copied from the grad-cam repository. Chooses the k most likely logits for each feature and creates their labels
    :param scores: scores for each feature
    :param k: Number of k-most-likely regions to create labels for
    :param labels: Array of all classes in the dataset
    :return: an array of array containing the k most likely labels and their scores
    """
    top_categories = np.argsort(scores, axis=1)[:, ::-1][:, :k]
    label_k = []
    for c in range(top_categories.shape[0]):
        categories = top_categories[c, :]
        clabels = []
        for category in categories:
            score = scores[c, category]
            if score >= 0.01:
                temp_label = f"{labels[category]}:{score:.2f}"
                clabels.append(temp_label)
        label_k.append("\n".join(clabels))
    return label_k


def adv_func(selected_attack, images, labels, model, bounds, epsilon, device="cpu"):
    """
    This function generates adversarial samples for a given method
    :param selected_attack: index of the attack to be used from [PGD, LinfDeepFoolAttack, LinfBasicIterativeAttack,
                     L2CarliniWagnerAttack, SpatialAttack, EADAttack, FGSM]
    :param images: one or multiple images to generate adversarials for
    :param labels: correct labels of the images
    :param model: model to create adversarials for
    :param bounds: bounds of the values for each pixel
    :param epsilon: the epsilon limiting the disturbances where applicable
    :param device: device on which to run the attack
    :return: The clipped adversarial, is_adv if the clipped adversarial causes a misclassification, the name of the adversarial method used, the minimal epsilon on a 0.01 level
    """
    methods_array = [PGD, LinfDeepFoolAttack, LinfBasicIterativeAttack,
                     L2CarliniWagnerAttack, SpatialAttack, EADAttack, FGSM]
    images = images.unsqueeze(dim=1)
    fool_model = fb.PyTorchModel(model, bounds, device)
    model.eval()
    # Special case for ead since it requires more parameters and larger epsilons
    if selected_attack == 5:
        attack = methods_array[selected_attack](initial_const=0.01, steps=1000, regularization=0.01,
                                                binary_search_steps=9, initial_stepsize=0.01, abort_early=False)
        ead_epsilon = 1
        raw, clipped, is_adv = attack(fool_model, images, labels.unsqueeze(dim=0), epsilons=ead_epsilon)
        # A speed-up for finding adversarial by iteratively refining the search, could be made more compact
        while not is_adv:
            print("Search Epsilon = " + str(ead_epsilon) + " did not work. Trying again with Epsilon = " + str(
                ead_epsilon + 10.0 - ead_epsilon % 10) + ". Sanity check: " + str(is_adv.item()))
            ead_epsilon = ead_epsilon - ead_epsilon % 10
            ead_epsilon = ead_epsilon + 10.0
            raw, clipped, is_adv = attack(fool_model, images, labels.unsqueeze(dim=0), epsilons=ead_epsilon)
            #  if epsilon is larger than 100 the image is likely too disturbed
            if ead_epsilon > 100:
                print("Attack unsuccessful since epsilon<=100 did not work")
                print(attack.__class__.__name__)
                return clipped, is_adv, attack.__class__.__name__, ead_epsilon
        print(f"Epsilon = {ead_epsilon} worked, refining search...")
        ead_epsilon -= 10.0
        if ead_epsilon <= 0:
            ead_epsilon = 1.0
        is_adv = torch.tensor(False)
        while not is_adv:
            print("Search Epsilon = " + str(ead_epsilon) + " did not work. Trying again with Epsilon = " + str(
                ead_epsilon + 1.0) + ". Sanity check: " + str(is_adv.item()))
            ead_epsilon = ead_epsilon + 1.0
            raw, clipped, is_adv = attack(fool_model, images, labels.unsqueeze(dim=0), epsilons=ead_epsilon)
        print(f"Epsilon = {ead_epsilon} worked, refining search...")
        ead_epsilon -= 1.0
        if ead_epsilon <= 0:
            ead_epsilon = 0.1
        is_adv = torch.tensor(False)
        while not is_adv:
            print("Search Epsilon = " + str(ead_epsilon) + " did not work. Trying again with Epsilon = " + str(
                round(ead_epsilon + 0.1, 1)) + ". Sanity check: " + str(is_adv.item()))
            ead_epsilon = round(ead_epsilon + 0.1, 1)
            raw, clipped, is_adv = attack(fool_model, images, labels.unsqueeze(dim=0), epsilons=ead_epsilon)
        print(f"Epsilon = {ead_epsilon} worked, refining search...")
        ead_epsilon -= 0.1
        if ead_epsilon <= 0:
            ead_epsilon = 0.01
        is_adv = torch.tensor(False)
        while not is_adv:
            print("Search Epsilon = " + str(ead_epsilon) + " did not work. Trying again with Epsilon = " + str(
                round(ead_epsilon + 0.01, 2)) + ". Sanity check: " + str(is_adv.item()))
            ead_epsilon = round(ead_epsilon + 0.01, 2)
            raw, clipped, is_adv = attack(fool_model, images, labels.unsqueeze(dim=0), epsilons=ead_epsilon)
        print(attack.__class__.__name__)
        return clipped, is_adv, attack.__class__.__name__, ead_epsilon
    else:
        # A speed-up for finding adversarial by iteratively refining the search, could be made more compact
        attack = methods_array[selected_attack]()
        raw, clipped, is_adv = attack(fool_model, images, labels.unsqueeze(dim=0), epsilons=epsilon)
        if not is_adv:
            while True:
                # if epsilon exceeds 10 the disturbance is too high and the attack is considered as failed
                if epsilon >= 10.0:
                    print("Epsilon is greater than 10, cancelling search...")
                    break
                print("Rough search Epsilon = " + str(epsilon) + " did not work. Trying again with Epsilon = " + str(
                    round(epsilon + 0.1, 1)) + ". Sanity check: " + str(is_adv.item()))
                epsilon += 0.1
                epsilon = round(epsilon, 1)
                raw, clipped, is_adv = attack(fool_model, images, labels.unsqueeze(dim=0),
                                              epsilons=epsilon)
                if is_adv.item():
                    print("Rough search Epsilon = " + str(
                        epsilon) + " worked, beginning fine search. Sanity check: " + str(is_adv.item()))
                    epsilon -= 0.1
                    epsilon = round(epsilon, 1)
                    if epsilon < 0.01:
                        epsilon = 0.01
                    is_adv = torch.tensor(False).to(device)
                    break
            while not is_adv:
                if epsilon >= 1.0:
                    print("Attack failed since epsilon is greater 1.0 and exceeds bounds.")
                    break
                print("Fine search Epsilon = " + str(epsilon) + " did not work. Trying again with Epsilon = " + str(
                    round(epsilon + 0.01, 2)) + ". Sanity check: " + str(is_adv.item()))
                epsilon += 0.01
                epsilon = round(epsilon, 2)
                raw, clipped, is_adv = attack(fool_model, images, labels.unsqueeze(dim=0),
                                              epsilons=epsilon)
        print(attack.__class__.__name__)
        return clipped, is_adv, attack.__class__.__name__, epsilon


##Preparing the data
model = FashionMNISTModelV1(input_shape=1, hidden_units=128, output_shape=10, dropout=0.5)

MODEL_NAME = "FashionMNISTModelStateDictPool.pth"

model.load_state_dict(torch.load(MODEL_NAME, map_location=device))
model.to(device)

train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

## Replacing / with _ in classnames since it interferes with the saving of the images
class_names = train_data.classes
for i in range(len(class_names)):
    class_names[i] = class_names[i].replace(" ", "_")
    class_names[i] = class_names[i].replace("/", "_")

# Manual seed in order to ensure reproducibility
manual_seed(931)
if torch.accelerator.is_available():
    torch.cuda.manual_seed(931)

##Sets some parameters before the loop
model.to(device)
model.eval()
target_layers = [model.layer1, model.layer4, model.layer9, model.layer12, model.layer17, model.layer20, model.layer23]
bounds = (0, 1)
classification_layer = model.layer29.to(device)

# Selects random classes and random images belonging to each class which are all classified correctly
test_pairs = []
selected_labels = []
# sets the number of classes to select
number_of_classes = 3
# sets the number of images per class
number_of_images = 10
# Increase or decrease
while len(selected_labels) < number_of_classes:
    idx = torch.randint(10000, ()).item()
    test_image, test_label = test_data[idx]
    if test_label not in selected_labels:
        if torch.softmax(model(test_image.unsqueeze(dim=0).to(device)), dim=1).argmax(dim=1) == test_label:
            test_pairs.append((test_image, test_label))
            selected_labels.append(test_label)
            chosen_label = test_label
            found_test_pairs = 1
            while found_test_pairs < number_of_images:
                idx = torch.randint(10000, ()).item()
                test_image, test_label = test_data[idx]
                if test_label == chosen_label:
                    if torch.softmax(model(test_image.unsqueeze(dim=0).to(device)), dim=1).argmax(dim=1) == test_label:
                        test_pairs.append((test_image, test_label))
                        found_test_pairs += 1

# Main Loop, first saves every unedited image, then calculates adversarial, saves it as is,
# then applies every cam for the new adversarial label and saves each, and finally applies the cam to each image without adversarial with their true label
# Initializes the directory to save the results to and the csv file a with some data for each result
result_dir = f"results"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
with open(f"results/result_data.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Method", "Percentage of change", "Absolute change of Visualization", "Absolute change of Adversarial", "Median of change", "Mean of change",
                     "Upper quartile of change", "Lower quartile of change", "Upper Whisker", "Lower Whisker",
                     "Pearson Correlation",
                     "p-value", "Null Hypothesis"])
#counts the number of times a label already appeared in order for each one to have a unique name like sandal_3 or shirt_9
label_itr = {}
# A dictionary containing all the vectors with the differences of each result
all_flat_vals = {}
for image, label in test_pairs:
    # Increases counter in order for each image to have its own unique associated number
    if class_names[label] not in list(label_itr.keys()):
        label_itr[class_names[label]] = 1
    else:
        label_itr[class_names[label]] = label_itr[class_names[label]] + 1

    #Visualizaes the original image and saves it
    image = image.to(device)
    label_tensor = torch.tensor(label, dtype=torch.long).to(device)
    plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
    plt.axis(False)
    plt.title("\n".join(wrap("Non-adversarial, classified as " + class_names[label], 60)))
    true_dir = f"results/no_adversarial/no_cam"
    if not os.path.exists(true_dir):
        os.makedirs(true_dir)
    plt.savefig(
        f"results/no_adversarial/no_cam/no_cam_on_unchanged_{class_names[label]}_{label_itr[class_names[label]]}.jpg",
        dpi=300,
        bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # calculates the visualizations for the current original image and saves them
    orig_cams = []
    for j in range(16):
        if j != 15:
            visualization, cam_name, raw_cam = grad_cam_func(target_layers, j, model, image, class_names, label,
                                                             classification_layer, device)
            orig_cams.append(raw_cam)
        else:
            visualization, cam_name = grad_cam_func(target_layers, j, model, image, class_names, label,
                                                    classification_layer, device)
        plt.imshow(visualization)
        plt.axis(False)
        plt.title("\n".join(wrap(f"Labeled as {class_names[label]}", 60)))
        cam_dir = f"results/no_adversarial/{cam_name}"
        if not os.path.exists(cam_dir):
            os.makedirs(cam_dir)
        plt.savefig(
            f"results/no_adversarial/{cam_name}/{cam_name}_without_adv_on_{class_names[label]}_{label_itr[class_names[label]]}.jpg",
            dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    # Generates adversarials for each image, the visualizations for each image and some corresponding data
    for i in range(7):

        #generates adversarial
        epsilon = 0.01
        adversarial, is_adv, adv_name, epsilon = adv_func(i, image, label_tensor, model, bounds, epsilon, device)

        # Saves the adversarial image
        plt.imshow(adversarial.squeeze().cpu().numpy(), cmap='gray')
        plt.axis(False)
        plt.title("\n".join(wrap(
            f"Adversarial: {is_adv.item()}, classified as {class_names[torch.softmax(model(adversarial.to(device)), dim=1).argmax(dim=1)]} with epsilon: {epsilon}",
            60)))
        adv_dir = f"results/{adv_name}/no_cam"
        if not os.path.exists(adv_dir):
            os.makedirs(adv_dir)
        plt.savefig(
            f"results/{adv_name}/no_cam/no_cam_with_{adv_name}_on_{class_names[label]}_{label_itr[class_names[label]]}.jpg",
            dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # calculates disturbance of adversarial
        adv_diff = torch.abs(torch.sub(image, adversarial)).squeeze().cpu()

        # Saves the disturbances as an image
        plt.imshow(adv_diff.numpy(), cmap='gray')
        plt.axis(False)
        plt.title("\n".join(wrap(f"Difference {adv_name} on original {class_names[label]}", 60)))
        plt.savefig(
            f"results/{adv_name}/no_cam/no_cam_difference_of_{adv_name}_on_{class_names[label]}_{label_itr[class_names[label]]}.jpg",
            dpi=300,
            bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # Stores all differences of the visualizations and the visualizaiton mathod name for each respective difference
        diff_arr = []
        cam_name_arr = []

        # creates and initializes a csv file for storing more data about the differences
        csvdir = f"results/{adv_name}/csvs"
        if not os.path.exists(csvdir):
            os.makedirs(csvdir)
        with open(f"results/{adv_name}/csvs/{adv_name}_on_{class_names[label]}_{label_itr[class_names[label]]}.csv",
                  "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Method", "Percentage of change", "Absolute change of Visualization", "Absolute change of Adversarial", "Median of change", "Mean of change",
                             "Upper quartile of change", "Lower quartile of change", "Upper Whisker", "Lower Whisker",
                             "Pearson Correlation",
                             "p-value", "Null Hypothesis"])
        # array storing all absolute difference sums
        abs_diffs = []
        # generates a visualization for each visualization method on the current adversarial
        for j in range(16):
            if j != 15:
                # creates the visualization
                visualization, cam_name, raw_cam = grad_cam_func(target_layers, j, model,
                                                                 adversarial.squeeze().unsqueeze(dim=0),
                                                                 class_names,
                                                                 torch.softmax(model(adversarial.to(device)),
                                                                               dim=1).argmax(dim=1),
                                                                 classification_layer, device)

                # calculates some data about the difference between this visualization and the original visualization
                diff = torch.abs(torch.sub(torch.tensor(raw_cam), torch.tensor(orig_cams[j])))
                percent_changed = torch.count_nonzero(diff) / torch.numel(diff)
                abs_sum = torch.sum(diff)
                adv_abs_sum = torch.sum(adv_diff)
                flat_values = torch.flatten(diff).numpy()
                median = np.median(flat_values)
                mean = np.mean(flat_values)
                upper_quartile = np.percentile(flat_values, 75)
                lower_quartile = np.percentile(flat_values, 25)
                inter_quartile_range = upper_quartile - lower_quartile
                upper_whisker = flat_values[flat_values <= upper_quartile + 1.5 * inter_quartile_range].max()
                lower_whisker = flat_values[flat_values >= lower_quartile - 1.5 * inter_quartile_range].min()
                diff_arr.append(flat_values)
                cam_name_arr.append(cam_name)
                abs_diffs.append(abs_sum)

                # The null Hypothesis is that both the adv change and CAM change correlated, which is why we calculate p = 1 - p
                pearsonc, p = pearsonr(adv_diff.flatten(), flat_values)
                p = 1 - p
                null_hypothesis = "Rejected" if p < 0.05 else "Accepted"

                # Creates a boxplot for the differences of all the pixels
                plt.boxplot(flat_values)
                plt.title("\n".join(
                    wrap(f"Boxplot of difference {adv_name} with {cam_name} on original {class_names[label]}", 60)))
                plt.xticks([1], [cam_name], rotation='vertical')
                dir = f"results/{adv_name}/{cam_name}"
                if not os.path.exists(dir):
                    os.makedirs(dir, exist_ok=True)
                plt.savefig(
                    f"results/{adv_name}/{cam_name}/Bplot_diff_{cam_name}_with_{adv_name}_on_{class_names[label]}_{label_itr[class_names[label]]}.jpg",
                    dpi=300, bbox_inches='tight', pad_inches=0.1)
                plt.close()

                # creates a dict in a dict in a dict to save the data in order for easier search
                if class_names[label] not in list(all_flat_vals.keys()):
                    all_flat_vals[class_names[label]] = {}
                if adv_name not in list(all_flat_vals[class_names[label]].keys()):
                    all_flat_vals[class_names[label]][adv_name] = {}
                if cam_name not in list(all_flat_vals[class_names[label]][adv_name].keys()):
                    all_flat_vals[class_names[label]][adv_name][cam_name] = []

                # saves the vector of differences in the dict
                name_id = class_names[label] + str(label_itr[class_names[label]])
                all_flat_vals[class_names[label]][adv_name][cam_name].append((flat_values, name_id))

                # Safes a heatmap of the difference of visualizations
                diff_vis = show_cam_on_image(image.permute(1, 2, 0).cpu().numpy(), diff.cpu().numpy(), use_rgb=True)
                plt.imshow(diff_vis)
                plt.axis(False)
                plt.title("\n".join(wrap(f"Difference {adv_name} for {cam_name} on original {class_names[label]}", 60)))
                plt.savefig(
                    f"results/{adv_name}/{cam_name}/Diff_{cam_name}_with_{adv_name}_on_{class_names[label]}_{label_itr[class_names[label]]}.jpg",
                    dpi=300, bbox_inches='tight', pad_inches=0.1)
                plt.close()

                # Saves the diffs themselves
                plt.imshow(diff.cpu(), cmap='gray')
                plt.axis(False)
                plt.title("\n".join(wrap(f"Grayscale Difference {adv_name} for {cam_name} on original {class_names[label]}", 60)))
                plt.savefig(
                    f"results/{adv_name}/{cam_name}/Gray_Diff_{cam_name}_with_{adv_name}_on_{class_names[label]}_{label_itr[class_names[label]]}.jpg",
                    dpi=300, bbox_inches='tight', pad_inches=0.1)
                plt.close()

                # Save the previously calculated data in both the csv  for the current cam and adversarial as well as the global csv
                with open(
                        f"results/{adv_name}/csvs/{adv_name}_on_{class_names[label]}_{label_itr[class_names[label]]}.csv",
                        "a",
                        newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(
                        [cam_name, percent_changed.item(), abs_sum.item(), adv_abs_sum.item(), median, mean,
                         upper_quartile, lower_quartile, upper_whisker, lower_whisker, pearsonc, p, null_hypothesis])
                with open(f"results/result_data.csv", "a",
                          newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(
                        [f"{class_names[label]}_with_{adv_name}_on_{cam_name}_{label_itr[class_names[label]]}", percent_changed.item(), abs_sum.item(),
                         adv_abs_sum.item(), median, mean,
                         upper_quartile, lower_quartile, upper_whisker, lower_whisker, pearsonc, p, null_hypothesis])
            else:
                # Skips alle the additional data collecting for Deep Feature Factorization since it is a fundamentally different method
                visualization, cam_name = grad_cam_func(target_layers, j, model,
                                                        adversarial.squeeze().unsqueeze(dim=0),
                                                        class_names,
                                                        torch.softmax(model(adversarial.to(device)),
                                                                      dim=1).argmax(dim=1),
                                                        classification_layer, device)

            # Saves the visualization of the adversarial
            plt.imshow(visualization)
            plt.axis(False)
            plt.title("\n".join(wrap(
                f"Adversarial: {is_adv.item()}, classified as {class_names[torch.softmax(model(adversarial.to(device)), dim=1).argmax(dim=1)]} with epsilon: {epsilon}",
                60)))
            dir = f"results/{adv_name}/{cam_name}"
            if not os.path.exists(dir):
                os.makedirs(dir, exist_ok=True)
            plt.savefig(
                f"results/{adv_name}/{cam_name}/{cam_name}_with_{adv_name}_on_{class_names[label]}_{label_itr[class_names[label]]}.jpg",
                dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()

        # Creates a boxplot with the differences of each visualization method on this adversarial
        plt.boxplot(diff_arr)
        plt.title(
            "\n".join(wrap(f"Boxplot of difference with {adv_name} for all CAMs on original {class_names[label]}", 60)))
        plt.xticks(np.arange(1, len(cam_name_arr) + 1), cam_name_arr, rotation='vertical')
        plt.tight_layout()
        bdir = f"results/{adv_name}/bplots"
        if not os.path.exists(bdir):
            os.makedirs(bdir, exist_ok=True)
        plt.savefig(
            f"results/{adv_name}/bplots/Bplot_diff_all_CAMs_with_{adv_name}_on_{class_names[label]}_{label_itr[class_names[label]]}.jpg",
            dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # Creates a barchart with the absolute differences of each visualization method on this adversarial
        plt.bar(cam_name_arr, abs_diffs)
        plt.title(
            "\n".join(
                wrap(f"Barchart of absolute difference with {adv_name} for all CAMs on original {class_names[label]}",
                     60)))
        plt.xticks(rotation='vertical')
        plt.tight_layout()
        bdir = f"results/{adv_name}/bcharts"
        if not os.path.exists(bdir):
            os.makedirs(bdir, exist_ok=True)
        plt.savefig(
            f"results/{adv_name}/bcharts/Barchart_diff_all_CAMs_with_{adv_name}_on_{class_names[label]}_{label_itr[class_names[label]]}.jpg",
            dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()

# Creates and saves bar chart for each adversarial- and visualization-method  for all adversarials of a class
for class_label in all_flat_vals:
    for adv_label in all_flat_vals[class_label]:
        for cam_label, vals in all_flat_vals[class_label][adv_label].items():
            all_diff_maps = []
            all_diff_names = []
            for diff_map, diff_name in vals:
                all_diff_maps.append(diff_map)
                all_diff_names.append(diff_name)
            plt.boxplot(all_diff_maps)
            plt.title(
                "\n".join(wrap(f"Boxplot of difference with {adv_label} for {cam_label} on all {class_label}", 60)))
            plt.xticks(np.arange(1, len(all_diff_names) + 1), all_diff_names, rotation='vertical')
            plt.tight_layout()
            plt.savefig(
                f"results/{adv_label}/{cam_label}/Bplot_for_all_{class_label}_with_{adv_label}_and_{cam_label}.jpg",
                dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()