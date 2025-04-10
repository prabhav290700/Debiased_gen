import numpy as np
import torch
from torchvision import transforms
from sklearn.cluster import KMeans
import clip
import torch.nn.functional as F
from torchmetrics.functional.pairwise import pairwise_cosine_similarity as cosine_similarity
# from transformers import CLIPProcessor, CLIPModel


def clip_preprocess(image):
    _, clip_preprocess = clip.load('ViT-B/32', device='cuda')

    preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                              clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                              clip_preprocess.transforms[4:])                                       # + skip convert PIL to tensor

    image_input = preprocess(image)
    return image_input


#r=2,scale=1 for h-space, r=20,sc=1.6 for x0
def clusterAndDelta(batch, proportion, clusters=2, r=2, scale=1, max_iters=100):
    epsi=0.0000001
    device = batch.device
    dtype = batch.dtype
    # print(batch.shape)
    batch = batch.clone().detach().cpu().numpy().astype('float32')
    n = batch.shape[0]
    shape = batch.shape
    bach = batch.reshape(n, -1)

    kmeans = KMeans(n_clusters=clusters, max_iter=max_iters, random_state=0,n_init=10)
    kmeans.fit(bach)
    initial_labels = kmeans.labels_
    # print(initial_labels)
    target_sizes = n*np.array(proportion)
    # print(target_sizes)
    initial_counts = np.bincount(initial_labels, minlength=clusters)
    ordered_clusters = np.argsort(initial_counts)
    # print(kmeans.cluster_centers_)
    centroids = kmeans.cluster_centers_ = kmeans.cluster_centers_[ordered_clusters]
    # print(kmeans.cluster_centers_)
    distances = kmeans.transform(bach)
    tags = np.full(n, -1)
    dist = np.full(n, -1)
    cluster_counts = {i: 0 for i in range(clusters)}
    
     # Assign points to maintain the desired proportions
    for i in range(n):
        centroid_dist = distances[i]
        for cluster_idx in np.argsort(centroid_dist):
            if cluster_counts[cluster_idx] < target_sizes[cluster_idx]:
                tags[i] = cluster_idx
                cluster_counts[cluster_idx] += 1
                dist[i]=centroid_dist[cluster_idx]
                break

    # (xi-zi)*gamma;  gamma=[1-r/(dist)]
    gamma=np.maximum(0,1-r/(dist+epsi)).reshape(n,-1)
    # tags = torch.tensor(tags, dtype=torch.long, device=device)
    
    delta=bach-centroids[tags]
    delta*=gamma*scale
    
    # centroids = torch.tensor(centroids, dtype=dtype, device=device)
    # centroids = centroids.view((clusters,) + shape[1:])
    # print(delta.shape)
    delta = torch.tensor(delta, dtype=dtype, device=device)
    delta = delta.view(shape)
    # model, clip_preprocess = clip.load(clipModel, device=device)
    # tokens = clip.tokenize(text).to(device)
    # text_features = model.encode_text(tokens).detach()
    
    return delta

    # dist=np.array(dist)
    # gamma = np.maximum(0, 1 - r / dist)
    # delta = centroids[tags] - bach
    # print(dist.shape)
    # delta *= gamma
    # centroids = torch.tensor(centroids, dtype=dtype, device=device).view((clusters,) + shape[1:])
    # delta = torch.tensor(delta, dtype=dtype, device=device).view(shape)
    
    # return tags, centroids, delta
@torch.enable_grad()
def clusterWithClip(batch, tags, clipModel, text):
    device = batch.device
    dtype = batch.dtype
    n = batch.shape[0]
    shape = batch.shape
    batch = batch.requires_grad_(True)
    # batch= (batch-batch.min())/(batch.max()-batch.min())
    
    # preprocess=transforms.Resize((224,224))
    # image_input=preprocess(batch)
    image_input = clip_preprocess(batch)

    tokens = clip.tokenize(text).to(device)
    # tokens = tokens.requires_grad_(True)

    # image_features = clipModel.encode_image(image_input).float()
    # text_features=text_features[tags]
    # similarity=torch.nn.functional.cosine_similarity(image_features, text_features)

    # logits_per_image, logits_per_text = clipModel(image_input, tokens)

    # print(logits_per_image)
    # print(tags)
    # 1/0
    # logits_desired = torch.tensor([logits_per_image[i, tags[i]] for i in range(n)], dtype=dtype, device=device)
    logits_desired = []
    for i in range(n):
        logit = clipModel(image_input[i].unsqueeze(0), tokens[tags[i]].unsqueeze(0))[0]
        # print(torch.autograd.grad(outputs=logit.mean(), inputs=batch)[0] )
        logits_desired.append(1. - logit/100)
        # print
    
    # logits_desired = torch.tensor(logits_desired, dtype=dtype, device=device)
    # print(logits_desired)
    
    # loss = (1. - logits_desired / 100).mean()
    loss = torch.stack(logits_desired).mean()
    # print(loss)

    # print(torch.autograd.grad(outputs=logits_desired[0].mean(), inputs=batch)[0] )

    # grads = torch.autograd.grad(outputs=loss, inputs=batch, grad_outputs=torch.ones_like(loss))[0] 
    grads = torch.autograd.grad(outputs=loss, inputs=batch)[0]  

    print(loss)
    
    return grads


@torch.no_grad()
def getTags(proportions,imgs,txts):
    #imgs,tags: embeddings mxd,nxd
    #proportions: list of fractions eg. [0.5,05] 
    assert len(proportions) == txts.shape[0]        
    tags=np.zeros(imgs.shape[0])
    target=np.array(proportions)*imgs.shape[0]
    target=target.astype(int)
    counts=[0]*txts.shape[0]
    scores=torch.nn.functional.cosine_similarity(imgs.unsqueeze(1), txts.unsqueeze(0), dim=2)
    max_values = scores.max(dim=1).values 
    sorted_indices = torch.argsort(max_values, descending=True)
    for i in sorted_indices:
        row = scores[i]
        row_vals = torch.argsort(row, descending=True)
        for j in row_vals:
            if counts[j]<target[j]:
                counts[j]+=1
                tags[i]=j
                break
    return tags











# @torch.enable_grad()
# def hClusterWithClip(h_current, hs_current, output_blocks, out_layer, emb, x, tags, 
#                      alpha, clipModel, text_features, scale=1.0):
#     h_pred = h_current
#     for module in output_blocks:
#         h_pred = torch.cat([h_pred, hs_current.pop()], dim=1)
#         h_pred = module(h_pred, emb)
#     h_pred = h_pred.type(x.dtype)
#     eps = out_layer(h_pred)
#     x0 = (x - ((1 - alpha).sqrt() * eps)) / (alpha.sqrt())

#     batch=x0
#     device = batch.device
#     dtype = batch.dtype
#     n = batch.shape[0]
#     shape = batch.shape
#     batch = batch.clone().requires_grad_(True)
#     batch= (batch-batch.min())/(batch.max()-batch.min())
    
#     preprocess=transforms.Resize((224,224))
#     image_input=preprocess(batch)
#     image_features = clipModel.encode_image(image_input).float()
#     text_features=text_features[tags]
#     similarity=torch.nn.functional.cosine_similarity(image_features, text_features)
    
#     loss=1-similarity
#     grads = torch.autograd.grad(outputs=loss, inputs=h_current, grad_outputs=torch.ones_like(loss))[0]
    
#     return h_current-grads* scale
    
    
    
    
    
    


    
    