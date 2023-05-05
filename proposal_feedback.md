Score: 5/8

Apologies for the delay in getting this feedback to you. This looks good so far, but I’d like a few more details in a few areas. You have a week from when you receive this until you need to submit a revised version. Please feel free to reach out with specific questions you have about this feedback, and please find a time to discuss it with your project mentor.

Essential goals:
1.	Are you planning to implement a diffusion model from scratch or train an existing implementation on the CIFAR data? If so, which one? If not, can you find at least one reference implementation that you might draw from if you get stuck? Basically, I’m fine either way, but it’s a lot more work to write it from scratch and debug the details than to use an existing implementation. If you want to focus on those implementation details, you might learn a lot from doing so, but it’s also going to give you less time to focus on the other analyses you hope to do.
2.	For FID, you can use this implementation: https://pytorch.org/ignite/generated/ignite.metrics.FID.html
3.	What pretrained GAN model will you use? Are you going to try to limit to the same single class of CIFAR-10 to constrain its generation? Or will you just do a general FID score evaluation?

Desired goals:
1.	Don’t frame this goal in terms of evaluation metrics you hope to achieve, but rather what you’ll change (i.e., which hyperparameters) about the diffusion model in terms of trying to improve the performance. Until we know what FID score your initial model gets, it’s hard to know if FID < 100 is a trivial or overly ambitious goal.
2.	You mention training a classifier to predict which class the image is from, but you don’t say that your diffusion model is going to do conditional generation. That is, are you planning to train the model to say “generate an image belonging to class X”? Tying this back to my earlier question, do you have an existing diffusion model in mind or are you going to be writing this from scratch?
3.	What will you use to compare your trained GAN against the diffusion model? FID score? Is this the same architecture as the pretrained GAN you plan to use for your essential goal?

Stretch goals:
1.	Your stretch goals should be more ambitious. The way I think you should frame this is, “if the desired goals go really smoothly, what would the three of us want to spend a week on to further our analysis?” Expanding to a higher-resolution dataset is reasonable, but only real challenge there is that it’ll require more computational resources; there’s not a major conceptual difference.
2.	Possible ideas for additional stretch goals:
-	If you decide not to use class-conditional generation in the earlier goals, add that to your diffusion and/or GAN models.
-	Try using model distillation or another method to reduce the size/runtime of your models
-	Try using explainability or visualization methods to understand what your models seems to learn about the input.
