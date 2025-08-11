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
from torch import nn, manual_seed
from torch.nn import ReLU, Conv2d, Linear, MaxPool2d, BatchNorm2d, Dropout, Flatten, AdaptiveAvgPool2d
from torchvision import datasets
from torchvision.transforms import ToTensor

##!!!!!Important: Matplotlib version <3.10.0 needed due to a mistake in grad-cam that returns an error on show_factorization_on_image
if version.parse(matplotlib.__version__) >= version.parse("3.9.0"):
    raise ImportError(
        f"Matplotlib<3.9.0 is required due to grad-cam, but found version == {matplotlib.__version__}"
    )
# Little Workaround to get ROCm to work with my "older" rx 6750 xt, since I else-wise get a HIP error for unsupported functions
os.putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else torch.device('cpu')
print(device)


##Defining the model and the functions generating adversarials and CAMs
class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape, dropout):
        super(FashionMNISTModelV1, self).__init__()
        # Block 1
        self.layer1 = Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1)
        self.layer2 = ReLU()
        self.layer3 = BatchNorm2d(hidden_units)
        self.layer4 = Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1)
        self.layer5 = ReLU()
        self.layer6 = BatchNorm2d(hidden_units)
        self.layer7 = MaxPool2d(2)
        self.layer8 = Dropout(dropout)
        # Block 2
        self.layer9 = Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1)
        self.layer10 = ReLU()
        self.layer11 = BatchNorm2d(hidden_units)
        self.layer12 = Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1)
        self.layer13 = ReLU()
        self.layer14 = BatchNorm2d(hidden_units)
        self.layer15 = MaxPool2d(2)
        self.layer16 = Dropout(dropout)
        # Block 3
        self.layer17 = Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1)
        self.layer18 = ReLU()
        self.layer19 = BatchNorm2d(hidden_units)
        self.layer20 = Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1)
        self.layer21 = ReLU()
        self.layer22 = BatchNorm2d(hidden_units)
        self.layer23 = Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1)
        self.layer24 = ReLU()
        self.layer25 = BatchNorm2d(hidden_units)
        self.layer26 = Dropout(dropout)
        # Global average pooling
        self.layer27 = AdaptiveAvgPool2d(1)
        self.layer28 = Flatten()
        self.layer29 = Linear(hidden_units, output_shape)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        x = self.layer18(x)
        x = self.layer19(x)
        x = self.layer20(x)
        x = self.layer21(x)
        x = self.layer22(x)
        x = self.layer23(x)
        x = self.layer24(x)
        x = self.layer25(x)
        x = self.layer26(x)
        x = self.layer27(x)
        x = self.layer28(x)
        x = self.layer29(x)
        return x


def grad_cam_func(targets, selected_cam, model, input_img, classes, label, classifier, device="cpu"):
    model.eval()
    methods_array = [GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus,
                     XGradCAM, AblationCAM, ScoreCAM, EigenCAM, EigenGradCAM,
                     LayerCAM, FullGrad, KPCA_CAM,
                     FEM, ShapleyCAM, FinerCAM]
    if selected_cam == 15:
        return deep_feature_factorization(targets, model, input_img, classes, label, classifier, device)
    else:
        if selected_cam == 13:
            last_layer = targets[-1]
            targets = [last_layer]
        method = methods_array[selected_cam](model=model, target_layers=targets)
        transformed_image = input_img.permute(1, 2, 0)
        cam = method(input_tensor=input_img.unsqueeze(dim=1))
        cam = cam[0, :]
        visualization = show_cam_on_image(transformed_image.cpu().numpy(), cam, use_rgb=True)
        print(method.__class__.__name__)
        return visualization, method.__class__.__name__, cam


def deep_feature_factorization(targets, model, input_img, classes, label, classifier_layer, device="cpu"):
    model.eval()
    ##Once again, some layers need to be moved to the cpu due to an error in the grad-cam library
    classifier_layer = classifier_layer.to("cpu")
    model = model.to("cpu")
    input_img = input_img.to("cpu")
    dff = DeepFeatureFactorization(model=model, target_layer=targets[len(targets) - 1].to("cpu"),
                                   computation_on_concepts=classifier_layer)
    transformed_image = input_img.permute(1, 2, 0)
    concepts, batch_explanations, concept_outputs = dff(input_img.unsqueeze(dim=1), len(classes))
    concept_outputs = torch.softmax(input=torch.from_numpy(concept_outputs), dim=-1).numpy()
    concept_label = create_dff_labels(concept_outputs, 2, class_names)
    transformed_image = torch.tensor(
        cv2.resize(transformed_image.numpy(), (224, 224), interpolation=cv2.INTER_LINEAR)).unsqueeze(dim=2)
    resized_explanations = []
    for exp in batch_explanations[0]:
        resized_explanations.append(cv2.resize(exp, (224, 224), interpolation=cv2.INTER_LINEAR))
    resized_explanations = np.array(resized_explanations)
    visualization = show_factorization_on_image(transformed_image.cpu().numpy(), resized_explanations, image_weight=0.3,
                                                concept_labels=concept_label)
    print(dff.__class__.__name__)
    targets[len(targets) - 1] = targets[len(targets) - 1].to(device)
    classifier_layer = classifier_layer.to(device)
    model = model.to(device)
    input_img = input_img.to(device)
    return visualization, dff.__class__.__name__


def create_dff_labels(scores, k, labels):
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
    methods_array = [PGD, LinfDeepFoolAttack, LinfBasicIterativeAttack,
                     L2CarliniWagnerAttack, SpatialAttack, EADAttack, FGSM]
    images = images.unsqueeze(dim=1)
    fool_model = fb.PyTorchModel(model, bounds, device)
    model.eval()
    if selected_attack == 5:
        attack = methods_array[selected_attack](initial_const=0.01, steps=1000, regularization=0.01,
                                                binary_search_steps=9, initial_stepsize=0.01, abort_early=False)
        ead_epsilon = 1
        raw, clipped, is_adv = attack(fool_model, images, labels.unsqueeze(dim=0), epsilons=ead_epsilon)
        while not is_adv:
            print("Search Epsilon = " + str(ead_epsilon) + " did not work. Trying again with Epsilon = " + str(
                ead_epsilon + 10.0 - ead_epsilon % 10) + ". Sanity check: " + str(is_adv.item()))
            ead_epsilon = ead_epsilon - ead_epsilon % 10
            ead_epsilon = ead_epsilon + 10.0
            raw, clipped, is_adv = attack(fool_model, images, labels.unsqueeze(dim=0), epsilons=ead_epsilon)
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
        attack = methods_array[selected_attack]()
        raw, clipped, is_adv = attack(fool_model, images, labels.unsqueeze(dim=0), epsilons=epsilon)
        if not is_adv:
            while True:
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


##Preparing data
model = FashionMNISTModelV1(input_shape=1, hidden_units=128, output_shape=10, dropout=0.5)

MODEL_NAME = "FashionMNISTModelStateDictPool.pth"

model.load_state_dict(torch.load(MODEL_NAME, map_location=device))
model.to(device)

train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

class_names = train_data.classes
for i in range(len(class_names)):
    class_names[i] = class_names[i].replace(" ", "_")
    class_names[i] = class_names[i].replace("/", "_")

# Getting the sets of data on which to generate the CAMs and Adversarials
manual_seed(931)
if torch.accelerator.is_available():
    torch.cuda.manual_seed(931)
test_pairs = []

##Setup before the loop
model.to(device)
model.eval()
target_layers = [model.layer1, model.layer4, model.layer9, model.layer12, model.layer17, model.layer20, model.layer23]
bounds = (0, 1)
classification_layer = model.layer29.to(device)
'''
while len(test_pairs) < 10:
    idx = torch.randint(10000,()).item()
    test_image, test_label = test_data[idx]
    if len(test_pairs) > 0:
        duplicate = False
        for i, l in test_pairs:
            if test_label == l:
                duplicate = True
                break
        if not duplicate:
            if torch.softmax(model(test_image.unsqueeze(dim=0).to(device)), dim=1).argmax(dim=1) == test_label:
                test_pairs.append((test_image, test_label))
    else:
        if torch.softmax(model(test_image.unsqueeze(dim=0).to(device)), dim=1).argmax(dim=1) == test_label:
            test_pairs.append((test_image, test_label))
'''
selected_labels = []
while len(selected_labels) < 3:
    idx = torch.randint(10000, ()).item()
    test_image, test_label = test_data[idx]
    if test_label not in selected_labels:
        if torch.softmax(model(test_image.unsqueeze(dim=0).to(device)), dim=1).argmax(dim=1) == test_label:
            test_pairs.append((test_image, test_label))
            selected_labels.append(test_label)
            chosen_label = test_label
            found_test_pairs = 1
            while found_test_pairs < 10:
                idx = torch.randint(10000, ()).item()
                test_image, test_label = test_data[idx]
                if test_label == chosen_label:
                    if torch.softmax(model(test_image.unsqueeze(dim=0).to(device)), dim=1).argmax(dim=1) == test_label:
                        test_pairs.append((test_image, test_label))
                        found_test_pairs += 1

# Main Loop, first saves every unedited image, then calculates adversarial, saves it as is,
# then applies every cam for the new adversarial label and saves each, and finally applies the cam to each image without adversarial with their true label
result_dir = f"results"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
with open(f"results/result_data.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Method", "Percentage of change", "Absolute change of Visualization", "Absolute change of Adversarial", "Median of change", "Mean of change",
                     "Upper quartile of change", "Lower quartile of change", "Upper Whisker", "Lower Whisker",
                     "Pearson Correlation",
                     "p-value", "Null Hypothesis"])
label_itr = {}
all_flat_vals = {}
for image, label in test_pairs:
    if class_names[label] not in list(label_itr.keys()):
        label_itr[class_names[label]] = 1
    else:
        label_itr[class_names[label]] = label_itr[class_names[label]] + 1
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
    for i in range(7):
        epsilon = 0.01
        adversarial, is_adv, adv_name, epsilon = adv_func(i, image, label_tensor, model, bounds, epsilon, device)
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
        adv_diff = torch.abs(torch.sub(image, adversarial)).squeeze().cpu()
        plt.imshow(adv_diff.numpy(), cmap='gray')
        plt.axis(False)
        plt.title("\n".join(wrap(f"Difference {adv_name} on original {class_names[label]}", 60)))
        plt.savefig(
            f"results/{adv_name}/no_cam/no_cam_difference_of_{adv_name}_on_{class_names[label]}_{label_itr[class_names[label]]}.jpg",
            dpi=300,
            bbox_inches='tight', pad_inches=0.1)
        plt.close()
        diff_arr = []
        cam_name_arr = []
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
        abs_diffs = []
        for j in range(16):
            if j != 15:
                visualization, cam_name, raw_cam = grad_cam_func(target_layers, j, model,
                                                                 adversarial.squeeze().unsqueeze(dim=0),
                                                                 class_names,
                                                                 torch.softmax(model(adversarial.to(device)),
                                                                               dim=1).argmax(dim=1),
                                                                 classification_layer, device)
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
                if class_names[label] not in list(all_flat_vals.keys()):
                    all_flat_vals[class_names[label]] = {}
                if adv_name not in list(all_flat_vals[class_names[label]].keys()):
                    all_flat_vals[class_names[label]][adv_name] = {}
                if cam_name not in list(all_flat_vals[class_names[label]][adv_name].keys()):
                    all_flat_vals[class_names[label]][adv_name][cam_name] = []
                name_id = class_names[label] + str(label_itr[class_names[label]])
                all_flat_vals[class_names[label]][adv_name][cam_name].append((flat_values, name_id))
                diff_vis = show_cam_on_image(image.permute(1, 2, 0).cpu().numpy(), diff.cpu().numpy(), use_rgb=True)
                plt.imshow(diff_vis)
                plt.axis(False)
                plt.title("\n".join(wrap(f"Difference {adv_name} for {cam_name} on original {class_names[label]}", 60)))
                plt.savefig(
                    f"results/{adv_name}/{cam_name}/Diff_{cam_name}_with_{adv_name}_on_{class_names[label]}_{label_itr[class_names[label]]}.jpg",
                    dpi=300, bbox_inches='tight', pad_inches=0.1)
                plt.close()
                plt.imshow(diff.cpu(), cmap='gray')
                plt.axis(False)
                plt.title("\n".join(wrap(f"Grayscale Difference {adv_name} for {cam_name} on original {class_names[label]}", 60)))
                plt.savefig(
                    f"results/{adv_name}/{cam_name}/Gray_Diff_{cam_name}_with_{adv_name}_on_{class_names[label]}_{label_itr[class_names[label]]}.jpg",
                    dpi=300, bbox_inches='tight', pad_inches=0.1)
                plt.close()
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
                visualization, cam_name = grad_cam_func(target_layers, j, model,
                                                        adversarial.squeeze().unsqueeze(dim=0),
                                                        class_names,
                                                        torch.softmax(model(adversarial.to(device)),
                                                                      dim=1).argmax(dim=1),
                                                        classification_layer, device)
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