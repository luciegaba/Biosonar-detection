import matplotlib.pyplot as plt
def plot_train_val_accuracy(model_history):
    # Récupérer les données de précision de l'historique d'entraînement
    train_acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']

    # Créer une plage pour le nombre d'époques
    epochs = range(1, len(train_acc) + 1)

    # Tracer la précision de l'entraînement et de la validation
    plt.plot(epochs, train_acc, 'b', label="Précision de l'entraînement")
    plt.plot(epochs, val_acc, 'r', label="Précision de la validation")

    # Ajouter des titres et des légendes au graphique
    plt.title("Précision de l'entraînement et de la validation")
    plt.xlabel("Époques")
    plt.ylabel("Précision")
    plt.legend()

    # Afficher le graphique
    plt.show()
    
    
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_pred_probs, title='Courbe ROC'):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


