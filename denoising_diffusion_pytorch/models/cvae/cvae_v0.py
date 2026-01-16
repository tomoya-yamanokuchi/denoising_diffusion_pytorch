import torch
from torch import nn, optim
from torch.nn import functional as F


















class ConvCVAE(nn.Module):
    def __init__(self, input_channels, cond_channels, hidden_dim, z_dim):
        super(ConvCVAE, self).__init__()
        
        # Encoder
        self.encoder_conv1 = nn.Conv2d(input_channels + cond_channels, 32, kernel_size=4, stride=2, padding=1)
        self.encoder_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.encoder_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.encoder_fc1 = nn.Linear(128 * 4 * 4, hidden_dim)
        self.encoder_fc21 = nn.Linear(hidden_dim, z_dim)
        self.encoder_fc22 = nn.Linear(hidden_dim, z_dim)
        
        # Decoder
        self.decoder_fc1 = nn.Linear(z_dim + cond_channels * 32 * 32, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, 128 * 4 * 4)
        self.decoder_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.decoder_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.decoder_conv3 = nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1)
    
    def encode(self, x, c):
        c = c.expand(x.shape[0], -1, x.shape[2], x.shape[3])
        x = torch.cat([x, c], dim=1)
        h1 = F.relu(self.encoder_conv1(x))
        h2 = F.relu(self.encoder_conv2(h1))
        h3 = F.relu(self.encoder_conv3(h2))
        h3 = h3.view(h3.size(0), -1)
        h4 = F.relu(self.encoder_fc1(h3))
        return self.encoder_fc21(h4), self.encoder_fc22(h4)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        c = c.view(c.size(0), -1)
        z = torch.cat([z, c], dim=1)
        h1 = F.relu(self.decoder_fc1(z))
        h2 = F.relu(self.decoder_fc2(h1))
        h2 = h2.view(h2.size(0), 128, 4, 4)
        h3 = F.relu(self.decoder_conv1(h2))
        h4 = F.relu(self.decoder_conv2(h3))
        return torch.sigmoid(self.decoder_conv3(h4))
    
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Example usage:
# Define model parameters
input_channels = 3  # For RGB images
cond_channels = 3   # For RGB condition images
hidden_dim = 512
z_dim = 20

# Instantiate model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvCVAE(input_channels, cond_channels, hidden_dim, z_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)



for i in range(10000):

    # input = torch.randn(5,1,28, 28)
    input = torch.randn(4,3,64, 64).to(device)
    cond  = torch.randn(3,64, 64).to(device)

    output_image, mu ,log_var=  model(input,cond)

    import ipdb;ipdb.set_trace()

    bce_loss= F.binary_cross_entropy(output_image, input, reduction='sum') # calculate loss

    print(f"bce_loss:{bce_loss}")



# Training loop
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     train_loss = 0
#     for batch_idx, (data, condition) in enumerate(train_loader):
#         data = data.to(device)
#         condition = condition.to(device)
#         optimizer.zero_grad()
#         recon_batch, mu, logvar = model(data, condition)
#         loss = loss_function(recon_batch, data, mu, logvar)
#         loss.backward()
#         train_loss += loss.item()
#         optimizer.step()
    
#     print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset)}')
