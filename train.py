import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, device, epochs=5):
    """
    Esegue il training del detector di deepfake.
    """
    # Usiamo la CrossEntropy perché è una classificazione binaria (Real vs Fake)
    criterion = nn.CrossEntropyLoss()
    # Ottimizzatore Adam (standard e veloce)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train() # Mette la rete in modalità addestramento
    
    print("Inizio Addestramento...")
    for epoch in range(epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 1. Azzera i gradienti
            optimizer.zero_grad()
            
            # 2. Forward pass (previsione)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 3. Backward pass (apprendimento)
            loss.backward()
            optimizer.step()
            
            # Statistiche
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct_predictions / total_samples
        
        print(f"Epoca [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Accuratezza: {epoch_acc:.2f}%")
        
    print("✅ Addestramento Completato!")
    return model