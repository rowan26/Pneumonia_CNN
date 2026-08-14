import torch

def train_one_epoch(model, dataloader, loss_fn, optimizer) -> tuple[float, float]:
    """
    """

    model.train()
    total_loss=0
    correct_predictions=0
    total_images=0

    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs=model(images)
        loss=loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()

        predictions=outputs.argmax(dim=1)
        correct_predictions+=(predictions == labels).sum().item()
        total_images+=labels.size(0)

    loss_mean= total_loss / len(dataloader)
    accuracy=correct_predictions / total_images

    return loss_mean, accuracy


def evaluate_one_epoch(model, dataloader, loss_fn) -> tuple[float, float]:

    """Évalue le modèle sur une epoch complète, sans mettre à jour les poids.
    Retourne la loss moyenne et l'accuracy sur ce dataloader (généralement val).
    """

    model.eval()
    total_loss=0
    correct_predictions=0
    total_images=0

    with torch.no_grad():
        for images, labels in dataloader:

            outputs=model(images)
            loss=loss_fn(outputs, labels)

            total_loss+=loss.item()
            predictions=outputs.argmax(dim=1)

            correct_predictions+=(predictions == labels).sum().item()
            total_images+=labels.size(0)
    
    loss_mean= total_loss / len(dataloader)
    accuracy=correct_predictions / total_images
    
    return loss_mean, accuracy


def train_model(
        model,
        train_dataloader, 
        val_dataloader, 
        loss_fn, 
        optimizer, 
        num_epochs: int=10) -> dict:

    """Entraîne le modèle sur plusieurs epochs, en évaluant sur val après
    chaque epoch. Retourne l'historique complet des métriques, pour permettre
    le tracking et la comparaison entre runs.
    """

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(num_epochs):

        train_loss, train_accuracy=train_one_epoch(model,train_dataloader,loss_fn,optimizer)
        val_loss, val_accuracy=evaluate_one_epoch(model,val_dataloader,loss_fn)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"train_loss: {train_loss:.4f} train_acc: {train_accuracy:.4f} | "
              f"val_loss: {val_loss:.4f} val_acc: {val_accuracy:.4f}")

    return history