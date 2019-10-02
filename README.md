# ACVAEGAN
A novel approach of imposing a condition on VAEGAN through the use of an auxiliary classifier.

## Description
- Goal: To build a Conditional VAEGAN by employing an Auxiliary Classifier.
- Architecture:
![ACVAEGAN Architecture](https://github.com/pranavbudhwant/acvaegan/blob/master/architecture.jpg)
- Dataset: WikiArt Emotions<sup>[1]</sup> 
- Approaches: 
  1. Generate paintings conditioned on emotion (anger, fear, sadness, ..)
  2. Generate paintings conditioned on category (cubism, surrealism, minimalism, ..)
  3. Generate paintings conditioned on style (contemporary, modern, renaissance, ..)
  
## Plan
- ~Prepare dataset for the three approaches~
  - ~CSV files containing (image-id, emotion); (image-id, category); (image-id, style)~
- Auxiliary Classifier Architecture
  - Multilabel Classifier/Multiclass Classifier?
- Keras, Pytorch implementation
- Try on MNIST
 
## References
[1] WikiArt Emotions: An Annotated Dataset of Emotions Evoked by Art. Saif M. Mohammad and Svetlana Kiritchenko. In Proceedings of the 11th Edition of the Language Resources and Evaluation Conference (LREC-2018), May 2018, Miyazaki, Japan.

[2] Autoencoding beyond pixels using a learned similarity metric https://arxiv.org/abs/1512.09300

[3] Conditional Image Synthesis With Auxiliary Classifier GANs https://arxiv.org/abs/1610.09585

[4] Twin Auxiliary Classifiers GAN https://arxiv.org/abs/1907.02690

[5] The Emotional GAN: Priming AdversarialGeneration of Art with Emotion https://nips2017creativity.github.io/doc/The_Emotional_GAN.pdf

[6] CVAE-GAN: Fine-Grained Image Generation through Asymmetric Training https://arxiv.org/pdf/1703.10155.pdf

[7] Learning Structured Output Representationusing Deep Conditional Generative Models https://pdfs.semanticscholar.org/3f25/e17eb717e5894e0404ea634451332f85d287.pdf
