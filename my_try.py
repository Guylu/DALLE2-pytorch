import torch
from dalle2_pytorch import DiffusionPriorNetwork, DiffusionPrior

# get trained CLIP from step one

# setup prior network, which contains an autoregressive transformer

prior_network = DiffusionPriorNetwork(
    dim=50,
    depth=6,
    dim_head=64,
    heads=8
).to("cuda:0")


# diffusion prior network, which contains the CLIP and network (with transformer) above

diffusion_prior = DiffusionPrior(
    net=prior_network,
    clip=None,
    image_embed_dim=50,
    timesteps=100,
    cond_drop_prob=0.2,
    condition_on_text_encodings=False,
).to("cuda:0")

# mock data
a1 = torch.load("../Best_attn_embds_eval2.pt", map_location=torch.device('cpu'))

inter = a1[0].index.intersection(a1[1].index)
a1[0] = a1[0].loc[inter]
a1[1] = a1[1].loc[inter]
print(f'number of samples is {len(inter)}')

from sklearn.model_selection import train_test_split

text_train, text_test, images_train, images_test = train_test_split(a1[0].values,
                                                                    a1[1].values,
                                                                    test_size=0.33, random_state=42)

# feed text and images into diffusion prior network
batch_size = 500

trainset = torch.utils.data.TensorDataset(torch.tensor(text_train).to("cuda:0"), torch.tensor(images_train).to("cuda:0"))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
import torch.optim as optim

optimizer = optim.SGD(diffusion_prior.parameters(), lr=1e-3)

diffusion_prior.train()
diffusion_prior = diffusion_prior.to("cuda:0")
for epoch in range(1000):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        text, images = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        loss = diffusion_prior(text_embed=text, image_embed=images)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    if epoch % 5 == 0:  # print every 2000 mini-batches
        print(f'[{epoch + 1}] loss: {running_loss / 5:.8f}')
        running_loss = 0.0
        with torch.no_grad():
            loss = diffusion_prior(text_embed=torch.tensor(text_test).cuda(),
                                   image_embed=torch.tensor(images_test).cuda())
            print(f'        [{epoch + 1}] Test loss: {loss:.8f}')

print('Finished Training')
diffusion_prior.eval()
with torch.no_grad():
    loss = diffusion_prior(text_embed=torch.tensor(text_test).cuda(),
                           image_embed=torch.tensor(images_test).cuda())
print(loss)

text_embed = torch.tensor(text_test).to("cuda:1")
image_embed = torch.tensor(images_test).cuda()

pred = diffusion_prior.to("cuda:1").sample(text_embed, num_samples_per_batch=3, cond_scale=1).detach().cpu().numpy()
