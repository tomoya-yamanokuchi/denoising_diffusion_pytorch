import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderModule(nn.Module):
    def __init__(self, input_channels, output_channels, stride, kernel, pad):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel, padding=pad, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Encoder(nn.Module):
    def __init__(self, color_channels, pooling_kernels, n_neurons_in_middle_layer):
        self.n_neurons_in_middle_layer = n_neurons_in_middle_layer
        super().__init__()

        self.bottle = EncoderModule(color_channels, 32, stride=1, kernel=1, pad=0)
        self.m1 = EncoderModule(32, 64, stride=1, kernel=3, pad=1)
        self.m2 = EncoderModule(64, 128, stride=pooling_kernels[0], kernel=3, pad=1)
        self.m3 = EncoderModule(128, 128, stride=pooling_kernels[1], kernel=3, pad=1)

        # self.bottle = EncoderModule(color_channels, 4, stride=1, kernel=1, pad=0)
        # self.m1 = EncoderModule(4, 8, stride=1, kernel=3, pad=1)
        # self.m2 = EncoderModule(8, 16, stride=pooling_kernels[0], kernel=3, pad=1)
        # self.m3 = EncoderModule(16, 32, stride=pooling_kernels[1], kernel=3, pad=1)

    def forward(self, x ,cond):
        cond.expand(x.shape[0], -1, x.shape[2], x.shape[3])

        input_w_cond = torch.cat([x,cond], dim = 1)

        out = self.m3(self.m2(self.m1(self.bottle(input_w_cond))))
        return out.view(-1, self.n_neurons_in_middle_layer)


class DecoderModule(nn.Module):
    def __init__(self, input_channels, output_channels, stride, activation="relu"):
        super().__init__()
        self.convt = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=stride, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.bn(self.convt(x)))


class Decoder(nn.Module):
    def __init__(self, color_channels, pooling_kernels, decoder_input_size):
        self.decoder_input_size = decoder_input_size
        super().__init__()

        self.m1 = DecoderModule(128, 128, stride=1)
        self.m2 = DecoderModule(128, 64, stride=pooling_kernels[1])
        self.m3 = DecoderModule(64, 32, stride=pooling_kernels[0])
        self.bottle = DecoderModule(32, color_channels, stride=1, activation="sigmoid")

        # self.m1 = DecoderModule(32, 16, stride=1)
        # self.m2 = DecoderModule(16, 8, stride=pooling_kernels[1])
        # self.m3 = DecoderModule(8, 4, stride=pooling_kernels[0])
        # self.bottle = DecoderModule(4, color_channels, stride=1, activation="sigmoid")


    def forward(self, x,):
        # out = x.view(-1, 256, self.decoder_input_size, self.decoder_input_size)
        out = x.view(-1, 128, self.decoder_input_size, self.decoder_input_size)
        # out = x.view(-1, 32, self.decoder_input_size, self.decoder_input_size)
        out = self.m3(self.m2(self.m1(out)))
        return self.bottle(out)


class VAE_2dim_conv(nn.Module):
    def __init__(self, input_dim = 64, latent_dim = 32, image_channels = 3, cond_channels=3):
        super().__init__()

        ## latent features
        self.n_latent_features = latent_dim

        # resolution
        # mnist, fashion-mnist : 28 -> 14 -> 7
        # mnist, fashion-mnist : 64 -> 16 -> 8
        pooling_kernel = [4, 2]
        encoder_output_size = 8


        input_channels = image_channels+cond_channels

        # neurons int middle layer
        # n_neurons_middle_layer   = 32 * encoder_output_size*encoder_output_size
        # n_neurons_middle_layer   = 256 * encoder_output_size*encoder_output_size
        n_neurons_middle_layer   = 128 * encoder_output_size * encoder_output_size
        n_neurons_middle_layer_2 = 32  * encoder_output_size


        self.relu = nn.ReLU(inplace=True)

        # Encoder
        self.encoder = Encoder(input_channels, pooling_kernel, n_neurons_middle_layer)

        # Middle
        self.fc11  = nn.Linear(n_neurons_middle_layer, n_neurons_middle_layer_2)
        self.fc12  = nn.Linear(n_neurons_middle_layer_2, self.n_latent_features)
        self.fc21  = nn.Linear(n_neurons_middle_layer, n_neurons_middle_layer_2)
        self.fc22  = nn.Linear(n_neurons_middle_layer_2, self.n_latent_features)


        self.decoder_fc1 = nn.Linear(n_neurons_middle_layer+ cond_channels * input_dim * input_dim, n_neurons_middle_layer_2)
        self.decoder_fc2 = nn.Linear(n_neurons_middle_layer_2, n_neurons_middle_layer)
        self.decoder_fc3 = nn.Linear(self.n_latent_features, n_neurons_middle_layer)

        # Decoder
        self.decoder = Decoder(image_channels, pooling_kernel, encoder_output_size)


    def forward(self, x , cond, mode = "train"):

        # Encoder
        h_vec = self.encoder(x,cond)

        # reparametarization trick
        mu_         = self.relu(self.fc11(h_vec))
        mu          = self.fc12(mu_)

        log_var_    = self.relu(self.fc21(h_vec))
        log_var     = self.fc22(log_var_)


        if mode == "train":
            # 標準正規乱数を振る
            eps = torch.randn_like(torch.exp(log_var))
            # 潜在変数の計算 μ + σ・ε
            latent_vector = mu + torch.exp(log_var / 2) * eps

        elif mode == "eval":
            latent_vector = mu

        # decoder
        decoder_cond    = cond.view(cond.size(0), -1)
        latent_vector_  = self.decoder_fc3(latent_vector)
        decoder_input_1 = torch.cat([latent_vector_,decoder_cond], dim = 1)
        decoder_input_2 = self.relu(self.decoder_fc1(decoder_input_1))
        decoder_input_3 = self.relu(self.decoder_fc2(decoder_input_2))

        reconstruct_img = self.decoder(decoder_input_3)

        return reconstruct_img, mu, log_var, latent_vector


    def sample(self, latent_vector , cond):

        # decoder
        decoder_cond    = cond.view(cond.size(0), -1)
        latent_vector_  = self.decoder_fc3(latent_vector)
        decoder_input_1 = torch.cat([latent_vector_,decoder_cond], dim = 1)
        decoder_input_2 = self.relu(self.decoder_fc1(decoder_input_1))
        decoder_input_3 = self.relu(self.decoder_fc2(decoder_input_2))

        reconstruct_img = self.decoder(decoder_input_3)

        return reconstruct_img


if __name__ == "__main__":

    print("hello model")
    image_channels = 3
    latent_dim     = 64
    device = "cuda:0"

    model = VAE_2dim_conv(latent_dim=latent_dim,image_channels=image_channels,cond_channels=3)
    model.to(device)

    for i in range(10000):
        input = torch.randn(64,3,64, 64).to(device)
        cond   = torch.randn(64,3,64, 64).to(device)

        output_image, mu ,log_var,latent_vector =  model(input,cond)


        bce_loss= F.binary_cross_entropy(output_image, input, reduction='sum') # calculate loss

        print(f"bce_loss:{bce_loss}")

    # image_dim = 64
    # input = torch.randn(64, 64)
    # input_ = input.view(-1,64*64)

    # import ipdb;ipdb.set_trace()

    # Net = VAE_2dim(input_dim=input.shape[0],latent_dim=32)
    # out, mu, log_var, z = Net(input_)

    # np_output = out.to('cpu').detach().numpy().copy()
    # np_image = np.reshape(np_output, (64, 64))

    # print(f"latent_tensor{z.shape}")
    # print(f"output_tensor{out.shape}")

    import ipdb;ipdb.set_trace()