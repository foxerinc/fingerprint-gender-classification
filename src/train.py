import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score

from .preprocessing import convert_to_grayscale_and_enchance_image, image_preprocesing_for_LBP
from .feature_extraction import store_LBP_features, store_ridge_density_features, store_hand
from .utils import get_label_from_filename

def run_experiment():
    images_directory = r'D:\SEMESTER 7\RTI\Sidik Jari\SOCOFing\SOCOFing\TRAIN'
    image_files = sorted([os.path.join(images_directory, f) for f in os.listdir(images_directory) if os.path.isfile(os.path.join(images_directory, f))])
    image_labels = {filename: get_label_from_filename(filename) for filename in image_files}
    labels = [label for label in image_labels.values()]

    #Image pre-processing
    # Process each image file for LBP
    images_for_LBP = [image_preprocesing_for_LBP(img) for img in image_files]
    # Process each image file for ridge density
    images_for_ridge_density = [convert_to_grayscale_and_enchance_image(img) for img in image_files]

    # Features extraction
    arr_lbp_features = store_LBP_features(images_for_LBP)
    arr_ridge_density_features = store_ridge_density_features(images_for_ridge_density)
    arr_hand_location = store_hand(image_files)

    X_lbp = np.array(arr_lbp_features)
    X_ridge_density = np.array(arr_ridge_density_features)
    X_array_hand_location = np.array(arr_hand_location)
    y = np.array(labels)

    #Feature fussion
    # Example: Uncomment the desired fusion method
    #LBP + Hand location
    # vector_features = [np.concatenate((X_lbp[i], X_array_hand_location[i])) for i in range(len(X_lbp))]

    #LBP + Ridge Density
    # vector_features = [np.concatenate((X_lbp[i], X_ridge[i])) for i in range(len(X_lbp))]

    #LBP + Ridge Density + Hand Location
    vector_features = [np.concatenate((X_lbp[i], X_ridge[i], X_hand[i])) for i in range(len(X_lbp))]

    #Ridge Density + Hand Location
    # vector_features = [np.concatenate((X_ridge[i], X_hand[i])) for i in range(len(X_lbp))]

    vector_features = np.array(vector_features)

    #Split Data
    X_train, X_test, y_train, y_test = train_test_split(vector_features, y, test_size=0.3, random_state=100)

    svm_linear = SVC(kernel='linear', C=100, gamma='auto')
    
    kf = KFold(n_splits=10, shuffle=True, random_state=100)

    fold_accuracies = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    test_accuracies = []
    
    train_vold_size = dict()
    val_vold_size = dict()
    
    # Iterate through each fold and train the model
    for fold, (train_index, val_index) in enumerate(kf.split(X_train), 1):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        
            # Track the size of training and validation sets for each fold
        train_vold_size[fold] = len(train_index)
        val_vold_size[fold] = len(val_index)
        svm_linear.fit(X_train_fold, y_train_fold)
            
            # Calculate accuracy for the fold
        val_accuracy = svm_linear.score(X_val_fold, y_val_fold) * 100
        fold_accuracies.append(val_accuracy)
        print(f"Accuracy report - Fold {fold}: {val_accuracy}")
        
        
        #test check
        test_accuracy = svm_linear.score(X_test,y_test) * 100
        test_accuracies.append(test_accuracy)
        y_pred_test = svm_linear.predict(X_test)
        precision = precision_score(y_test,y_pred_test,zero_division=1)
        
            
            # Predict labels for the validation set
        y_pred_val = svm_linear.predict(X_val_fold)
            
            # Generate confusion matrix for the validation set
        cm = confusion_matrix(y_val_fold, y_pred_val)
        
        # Calculate additional metrics
        precision = precision_score(y_val_fold, y_pred_val, average='weighted',zero_division=1)
        recall = recall_score(y_val_fold, y_pred_val, average='weighted',zero_division=1)
        f1 = f1_score(y_val_fold, y_pred_val, average='weighted',zero_division=1)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")
            
            # Plot confusion matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - Fold {fold}")
        plt.xticks([0.5, 1.5], ["female", "male"])
        plt.yticks([0.5, 1.5], ["female", "male"])
        plt.show()
        plt.close()
        # Generate and print classification report
        report = classification_report(y_val_fold, y_pred_val)
        print(f"Classification Report - Fold {fold}:\n{report}")
        
        

        # Plot accuracy for each fold
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, k + 1), fold_accuracies, color='b')
    plt.title('Validation Accuracy for Each Fold')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.show()
    plt.close()
    
    
        # Print overall metrics
    print(f"Overall Accuracy: Rata-rata: {np.mean(fold_accuracies):.2f} ± Standar Deviasi: {np.std(fold_accuracies):.3f}")
    print(f"Overall Precision: Rata-rata: {np.mean(precision_scores):.2f} ± Standar Deviasi: {np.std(precision_scores):.3f}")
    print(f"Overall Recall: Rata-rata: {np.mean(recall_scores):.2f} ± Standar Deviasi: {np.std(recall_scores):.3f}")
    print(f"Overall F1-Score: Rata-rata: {np.mean(f1_scores):.2f} ± Standar Deviasi: {np.std(f1_scores):.3f}")