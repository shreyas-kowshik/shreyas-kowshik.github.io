---
title: "Community Bonding"
subtitle: "Holiday's on a close, work nearing"
layout: post
tags: [machine-learning,generative-models]
---

The three weeks of <b>The Community Bonding Period</b> have almost come to a close. It's time to start to write some serious code and get some models running. My experience with this period was mostly some interaction on Slack along with writing code. 

As I did not need many intricacies for the first two models under my proposal, I decided to begin work after talking to my mentor. The models that have been implemented till now are - `CycleGAN.jl` and `pix2pix.jl`. 

Let me explain the code flow and walk you through the impelmentation.

## CycleGAN.jl

`CycleGAN` is an implementation of [this](https://arxiv.org/pdf/1703.10593.pdf) paper.

The architecture is as follows : 
![cyclegan-architecture](/blog/figs/2019-5-23/cycle-gan-architecture.png)

Basically it learns the mapping between images from one domain to another. There is a generator and a discriminator, one each for the two domains. The problem is formulated as a normal GAN with constraints on the construction of images. We'll go in more detail on this as we discuss the loss functions.

[Here](https://github.com/shreyas-kowshik/CycleGAN.jl) is the link to the code.

<b> Loading The Dataset </b>

We use the `apples2oranges` dataset. Download and extract it to the `data` directory. Let's load our dataset.

```julia
function load_image(filename)
    img = load(filename)
    img = Float64.(channelview(img))
end

function load_dataset(path,imsize)
   imgs = []
   for r in readdir(path)
        img_path = string(path,r)
        push!(imgs,load_image(img_path))
    end
    reshape(hcat(imgs...),imsize,imsize,3,length(imgs))
end

# Load the dataset
dataA = load_dataset("../data/trainA/",256) |> gpu
dataB = load_dataset("../data/trainB/",256) |> gpu
```

<b> Building The Architectures </b>

The paper uses the `UNet` architecture for the generators and a simple sequentially downsampling discriminator. We avoid using `MaxPool` layer to avoid sparse gradients during the GAN training. 

[Here](https://github.com/shreyas-kowshik/CycleGAN.jl/blob/master/src/generator.jl) is the reference code to follow along.

A `UNet` is basically an encoder-decoder with skip connections in between. We define two modules, one for downsampling and one for upsampling : 

```julia
# Convolution And Downsample
ConvDown(in_chs,out_chs)... # Arguments are the input and output number of channels

# Convolution And Upsample
struct UNetUpBlock
    upsample
    conv_layer
end

UNetUpBlock(in_chs,out_chs)... # Arguments are the input and output nuumber of channels

function(u::UNetUpBlock)(x,bridge)
	# Upsample -> concatenate [up(x),bridge] -> convolution
end

struct UNet
    conv_down_blocks # Convolve And Downsample
    conv_blocks # Convolve
    up_blocks # Upsample, concatenate, convolve
end
```

The [discriminator](https://github.com/shreyas-kowshik/CycleGAN.jl/blob/master/src/discriminator.jl) is a sequence of convolutions with strides such that the image is spatially halved at each step. A `sigmoid` activation is appended at the end of the model to represent the probability that the input image is from the real distribution.

<b> Writing The Loss Functions </b>

The discriminator loss is the standard adversarial loss as in a normal GAN. It tires to classify the real image as real and the generated image as fake.

```
function dA_loss(a,b)
    """
    a : Image in domain A
    b : Image in domain B
    """
    # LABELS #
    real_labels = ones(1,BATCH_SIZE) |> gpu
    fake_labels = zeros(1,BATCH_SIZE) |> gpu

    fake_A = gen_B(b) # Fake image generated in domain A
    fake_A_prob = drop_first_two(dis_B(fake_A.data)) # Probability that generated image in domain A is real
    real_A_prob = drop_first_two(dis_B(a)) # Probability that original image in domain A is real

    dis_A_real_loss = ((real_A_prob .- real_labels).^2)
    dis_A_fake_loss = ((fake_A_prob .- fake_labels).^2)
    convert(Float32,0.5) * mean(dis_A_real_loss + dis_A_fake_loss)
end
```

The generator loss is a bit more interesting. Apart from the standard adversarial loss, to enforce the constraints on the structure of the output, a reconstruction and an identity loss is enforced. 

The model is designed for <b>unpaired</b> image to image translation. If only adversarial losses are used, a one-to-many mapping is possible for each image in domain A. One desires that only the relevant part of the input is translated. For instance, an image must only convert an apple to an orange while leaving the background unchanged. This constraint is enforced usnig the reconstruction loss. The generated domain B image when passed through the `B->A` generator must reconstruct the input. The identity loss is also enforced to complement this process. It basically states that for an input image of domain A to the `B->A` generator, the generator must behave like an identity function.

```julia
function g_loss(a,b)
    """
    a : Image in domain A
    b : Image in domain B
    """
    # LABELS #
    real_labels = ones(1,BATCH_SIZE) |> gpu
    fake_labels = zeros(1,BATCH_SIZE) |> gpu

    # Forward Propogation # 
    fake_B = gen_A(a) # Fake image generated in domain B
    fake_B_prob = dis_B(fake_B) # Probability that generated image in domain B is real
    real_B_prob = dis_B(b) # Probability that original image in domain B is real

    fake_A = gen_B(b) # Fake image generated in domain A
    fake_A_prob = drop_first_two(dis_A(fake_A)) # Probability that generated image in domain A is real
    real_A_prob = drop_first_two(dis_A(a)) # Probability that original image in domain A is real
    
    # Reconstructions #
    rec_A = gen_B(fake_B)
    rec_B = gen_A(fake_A)
    
    ### Generator Losses ###
    # For domain A->B  #
    gen_B_loss = mean((fake_B_prob .- real_labels).^2) # Adversarial loss
    rec_B_loss = mean(abs.(b .- rec_B)) # Reconstruction loss for domain B 
    
    # For domain B->A  #
    gen_A_loss = mean((fake_A_prob .- real_labels).^2) # Adversarial loss
    rec_A_loss = mean(abs.(a .- rec_A)) # Reconstrucion loss for domain A 

    # Identity losses 
    # gen_A should be identity if b is fed : ||gen_A(b) - b||
    idt_A_loss = mean(abs.(gen_A(b) .- b))
    # gen_B should be identity if a is fed : ||gen_B(a) - a||
    idt_B_loss = mean(abs.(gen_B(a) .- a))

    gen_A_loss + gen_B_loss + λ₁*rec_A_loss + λ₂*rec_B_loss  + λid*(λ₁*idt_A_loss + λ₂*idt_B_loss)
end
```

The model training is currently not complete as I did not have access to a good GPU machine until now. Work on that should start soon.

<b> Sampling from the generator </b>

After having trained our model, we would want to go out and convert an apple to an orange, right? We turn the generator network into `testmode` which ensures that batchnorm and other layers use their inference time properties.

```julia
function sampleA2B(X_A_test)
    """
    Samples new images in domain B
    X_A_test : N x C x H x W array - Test images in domain A
    """
    testmode!(gen_A)
    X_A_test = norm(X_A_test)
    X_B_generated = cpu(denorm(gen_A(X_A_test |> gpu)).data)
    testmode!(gen_A,false)
    imgs = []
    s = size(X_B_generated)
    for i in size(X_B_generated)[end]
       push!(imgs,colorview(RGB,reshape(X_B_generated[:,:,:,i],3,s[1],s[2])))
    end
    imgs
end

function test()
   # load test data
   dataA = load_dataset("../data/trainA/",256)[:,:,:,1:2] |> gpu
   out = sampleA2B(dataA)
   for (i,img) in enumerate(out)
        save("../sample/A_$i.png",img)
   end
end
```

That should complete all of the building blocks required to glue together a `CycleGAN` model. Let's now move onto the `pix2pix` network.

## pix2pix.jl

[Here](https://arxiv.org/pdf/1611.07004.pdf) is the link to the paper.

This model also solves the problem of image to image translation. However, the translations are paired up here. Thus each input image can correspond to only one output image. The concept is precisely a conditional GAN, with the input to the discriminator conditioned on the input image.

The [code](https://github.com/shreyas-kowshik/pix2pix.jl) for data loading and the architectures are almost the same here as in `CycleGAN.jl`. 

The difference lies in the loss function implementation. The discriminator loss is the general adversarial GAN loss with the input conditioned on the primary domain image. The generator loss consists of an adversarial loss along with a loss that weights proper reconstruction. It aims to minimise the difference between the generated output and the image in the target domain.

```julia
function d_loss(a,b)
    """
    a : Image in domain A
    b : Image in domain B
    """
    fake_B = gen(a |> gpu)
    fake_AB = cat(fake_B,a,dims=3) |> gpu

    fake_prob = dis(fake_AB)
    loss_D_fake = bce(fake_prob,fake_labels)

    real_AB =  cat(b,a,dims=3) |> gpu
    real_prob = dis(real_AB)
    loss_D_real = bce(real_prob,real_labels)

    0.5 * mean(loss_D_real .+ loss_D_fake)
end

function g_loss(a,b)
    """
    a : Image in domain A
    b : Image in domain B
    """
    fake_B = gen(a |> gpu)
    fake_AB = cat(fake_B,a,dims=3) |> gpu

    fake_prob = dis(fake_AB)

    loss_adv = mean(bce(fake_prob,real_labels))
    loss_L1 = mean(abs.(fake_B .- b)) 
    loss_adv + λ*loss_L1
end
```

The training intricacies are similar here as compared to the `CycleGAN` model.

### Conclusion

The two weeks of the <b>Community Bonding Period</b> turned out to be a great experience. I was able to read up literature on GANs, the papers and write some code. Besides, I also set out to understanding the math behind `Policy Gradient Algorithms`, which would turn out to be handy while debugging in the later stages of the coding period, wherein I shall be implementing some advanced algorithms of these types.