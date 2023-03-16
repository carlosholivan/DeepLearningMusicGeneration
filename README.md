# <span style="color:#9EB1FF; font-size:30.0pt">DEEP LEARNING FOR MUSIC GENERATION</span>


This repository is maintained by [**Carlos Hernández-Oliván**](https://carlosholivan.github.io/index.html)(carloshero@unizar.es) and it presents the State of the Art of Music Generation. Most of these references (previous to 2022) are included in the review paper ["Music Composition with Deep Learning: A Review"](#https://arxiv.org/abs/2108.12290). The authors of the paper want to thank Jürgen Schmidhuber for his suggestions.

[![License](https://img.shields.io/badge/license-Apache2.0-green)](./LICENSE)

Make a pull request if you want to contribute to this references list.

You can download a PDF version of this repo here: [README.pdf](AIMusicGeneration.pdf)

All the images belong to their corresponding authors.

## Table of Contents

1. [Algorithmic Composition](#algorithmic-composition)

    - [1992](#1992alg)

    - [Books](#books-alg)


2. [Neural Network Architectures](#neural-network-architectures)

3. [Deep Learning Models for Symbolic Music Generation](#deep-learning-music-generation)

    - [2023](#2023deep)
    - [2022](#2022deep)
    - [2021](#2021deep)
    - [2020](#2020deep)
    - [2019](#2019deep)
    - [2018](#2018deep)
    - [2017](#2017deep)
    - [2016](#2016deep)
    - [2015](#2015deep)
    - [2002](#2002deep)
    - [1990s](#1990deep)

    - [Books and Reviews](#books-reviews-deep)
      - [Books](#books-deep)
      - [Reviews](#reviews-deep)

4. [Deep Learning Models for Audio Music Generation](#deep-learning-audio-generation)

    - [2023](#2023audiodeep)
    - [2022](#2022audiodeep)
    - [2021](#2021audiodeep)
    - [2020](#2020audiodeep)
    - [2017](#2017audiodeep)

5. [Datasets](#datasets)

6. [Journals and Conferences](#journals)

7. [Authors](#authors)

8. [Research Groups and Labs](#labs)

10. [Apps for Music Generation with AI](#apps)

11. [Other Resources](#other-resources)



## <span id="algorithmic-composition" style="color:#9EB1FF; font-size:25.0pt">2. Algorithmic Composition</span>

### <span id="1992alg" style="color:#A8FF9E; font-size:20.0pt">1992</span>

#### <span id="harmonet" style="color:#FF9EC3; font-size:15.0pt">HARMONET</span>


Hild, H., Feulner, J., & Menzel, W. (1992). HARMONET: A neural net for harmonizing chorales in the style of JS Bach. In Advances in neural information processing systems (pp. 267-274). [Paper](https://proceedings.neurips.cc/paper/1991/file/a7aeed74714116f3b292a982238f83d2-Paper.pdf)


### <span id="books-alg" style="color:#A8FF9E; font-size:25.0pt">Books</span>

* Westergaard, P. (1959). Experimental Music. Composition with an Electronic Computer.

* Todd, P. M. (1989). A connectionist approach to algorithmic composition. Computer Music Journal, 13(4), 27-43.

* Cope, D. (2000). The algorithmic composer (Vol. 16). AR Editions, Inc..

* Nierhaus, G. (2009). Algorithmic composition: paradigms of automated music generation. Springer Science & Business Media.

* Müller, M. (2015). Fundamentals of music processing: Audio, analysis, algorithms, applications. Springer.

* McLean, A., & Dean, R. T. (Eds.). (2018). The Oxford handbook of algorithmic music. Oxford University Press.


## <span id="neural-network-architectures" style="color:#9EB1FF; font-size:25.0pt">2. Neural Network Architectures</span>

| NN Architecture | Year | Authors | Link to original paper | Slides |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Long Short-Term Memory (LSTM) | 1997 | Sepp Hochreiter, Jürgen Schmidhuber | http://www.bioinf.jku.at/publications/older/2604.pdf | [LSTM.pdf](Slides/LSTM_v1.pdf) |
| Convolutional Neural Network (CNN) | 1998 | Yann LeCun, Léon Bottou, YoshuaBengio, Patrick Haffner | http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf |  |
| Variational Auto Encoder (VAE) | 2013 | Diederik P. Kingma, Max Welling | https://arxiv.org/pdf/1312.6114.pdf |
| Generative Adversarial Networks (GAN) | 2014 | Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio | https://arxiv.org/pdf/1406.2661.pdf |  | 
| Transformer | 2017 | Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin | https://arxiv.org/pdf/1706.03762.pdf | |
| Diffusion Models | 2015 | Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, Surya Ganguli | https://arxiv.org/abs/1503.03585 | |


## <span id="deep-learning-music-generation" style="color:#9EB1FF; font-size:25.0pt">3. Deep Learning Models for Music Generation</span>

### <span id="2023deep" style="color:#A8FF9E; font-size:20.0pt">2023</span>

#### <span id="rl-chord" style="color:#FF9EC3; font-size:15.0pt">RL-Chord</span>

Ji, S., Yang, X., Luo, J., & Li, J. (2023). RL-Chord: CLSTM-Based Melody Harmonization Using Deep Reinforcement Learning. IEEE Transactions on Neural Networks and Learning Systems.

[Paper](https://ieeexplore.ieee.org/abstract/document/10063204)

#### <span id="figaro" style="color:#FF9EC3; font-size:15.0pt">FIGARO: Generating Symbolic Music with Fine-Grained Artistic Control</span>

von Rütte, D., Biggio, L., Kilcher, Y., & Hoffman, T. (2022). FIGARO: Generating Symbolic Music with Fine-Grained Artistic Control. Accepted ICLR 2023.

<img src="images/Figaro.png" width="100" height="150">

[Paper](https://arxiv.org/abs/2201.10936)

### <span id="2022deep" style="color:#A8FF9E; font-size:20.0pt">2022</span>

#### <span id="museformer" style="color:#FF9EC3; font-size:15.0pt">Museformer</span>

Yu, B., Lu, P., Wang, R., Hu, W., Tan, X., Ye, W., ... & Liu, T. Y. (2022). Museformer: Transformer with Fine-and Coarse-Grained Attention for Music Generation. NIPS 2022.

[Paper](https://openreview.net/forum?id=GFiqdZOm-Ei) [NIPS Presentation](https://nips.cc/virtual/2022/poster/54604)

#### <span id="bar-transformer" style="color:#FF9EC3; font-size:15.0pt">Bar Transformer</span>

Qin, Y., Xie, H., Ding, S., Tan, B., Li, Y., Zhao, B., & Ye, M. (2022). Bar transformer: a hierarchical model for learning long-term structure and generating impressive pop music. Applied Intelligence, 1-19.

[Paper](https://link.springer.com/article/10.1007/s10489-022-04049-3)

#### <span id="sympony-generation" style="color:#FF9EC3; font-size:15.0pt">Symphony Generation with Permutation Invariant Language Model</span>

Liu, J., Dong, Y., Cheng, Z., Zhang, X., Li, X., Yu, F., & Sun, M. (2022). Symphony Generation with Permutation Invariant Language Model. arXiv preprint arXiv:2205.05448.

<img src="images/Symphony Generation.png" width="300" height="120">

[Paper](http://128.84.4.34/abs/2205.05448) [Code](https://github.com/symphonynet/SymphonyNet) [Samples](https://symphonynet.github.io/)


#### <span id="theme-transformer" style="color:#FF9EC3; font-size:15.0pt">Theme Transfomer</span>

Shih, Y. J., Wu, S. L., Zalkow, F., Muller, M., & Yang, Y. H. (2022). Theme Transformer: Symbolic Music Generation with Theme-Conditioned Transformer. IEEE Transactions on Multimedia.

<img src="images/Theme Transformer.png" width="300" height="100">

[Paper](https://arxiv.org/abs/2111.04093) [GitHub](https://github.com/atosystem/ThemeTransformer)



### <span id="2021deep" style="color:#A8FF9E; font-size:20.0pt">2021</span>


#### <span id="compound-word" style="color:#FF9EC3; font-size:15.0pt">Compound Word Transformer</span>

Hsiao, W. Y., Liu, J. Y., Yeh, Y. C., & Yang, Y. H. (2021, May). Compound word transformer: Learning to compose full-song music over dynamic directed hypergraphs. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 35, No. 1, pp. 178-186).

[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/16091) [GitHub](https://github.com/YatingMusic/compound-word-transformer)

#### <span id="melody-lyrics-models" style="color:#FF9EC3; font-size:15.0pt">Melody Generation from Lyrics</span>

Yu, Y., Srivastava, A., & Canales, S. (2021). Conditional lstm-gan for melody generation from lyrics. ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM), 17(1), 1-20.

<img src="images/Melody Generation from Lyrics.jpg" width="300" height="200">

[Paper](https://dl.acm.org/doi/abs/10.1145/3424116)


#### <span id="diffusion-models" style="color:#FF9EC3; font-size:15.0pt">Music Generation with Diffusion Models</span>

Mittal, G., Engel, J., Hawthorne, C., & Simon, I. (2021). Symbolic music generation with diffusion models. arXiv preprint arXiv:2103.16091.

<img src="images/Music Generation with Diffusion Models.png" width="400" height="200">

[Paper](https://arxiv.org/abs/2103.16091) [GitHub](https://github.com/magenta/symbolic-music-diffusion)

### <span id="2020deep" style="color:#A8FF9E; font-size:20.0pt">2020</span>

#### <span id="pop-musc-transfomer" style="color:#FF9EC3; font-size:15.0pt">Pop Musc Transfomer</span>

Huang, Y. S., & Yang, Y. H. (2020, October). Pop music transformer: Beat-based modeling and generation of expressive pop piano compositions. In Proceedings of the 28th ACM International Conference on Multimedia (pp. 1180-1188).

[Paper](https://dl.acm.org/doi/abs/10.1145/3394171.3413671) [GitHub](https://github.com/YatingMusic/remi)


#### <span id="controllable-polyphonic" style="color:#FF9EC3; font-size:15.0pt">Controllable Polyphonic Music Generation</span>

Wang, Z., Wang, D., Zhang, Y., & Xia, G. (2020). Learning interpretable representation for controllable polyphonic music generation. arXiv preprint arXiv:2008.07122.

<img src="images/Controllable Polyphonic Music Generation.png" width="200" height="200">

[Paper](https://arxiv.org/abs/2008.07122) [Web](https://program.ismir2020.net/poster_5-05.html) [Video](https://www.youtube.com/watch?v=Sb6jXP_7dtE&t=28s&ab_channel=ISMIR2020)

#### <span id="mmm" style="color:#FF9EC3; font-size:15.0pt">MMM: Multitrack Music Generation</span>

Ens, J., & Pasquier, P. (2020). Mmm: Exploring conditional multi-track music generation with the transformer. arXiv preprint arXiv:2008.06048.

<img src="images/MMM Multitrack Music Generation.png" width="300" height="200">

[Paper](https://arxiv.org/abs/2008.06048) [Web](https://jeffreyjohnens.github.io/MMM/) [Colab](https://colab.research.google.com/drive/1xGZW3GP24HUsxnbebqfy1iCyYySQ64Vs?usp=sharing) [Github (AI Guru)](https://github.com/AI-Guru/MMM-JSB)

#### <span id="xl" style="color:#FF9EC3; font-size:15.0pt">Transformer-XL</span>

Wu, X., Wang, C., & Lei, Q. (2020). Transformer-XL Based Music Generation with Multiple Sequences of Time-valued Notes. arXiv preprint arXiv:2007.07244.

<img src="images/Transformer-XL.png" width="400" height="300">

[Paper](https://arxiv.org/abs/2007.07244)


#### <span id="transformer-vae" style="color:#FF9EC3; font-size:15.0pt">Transformer VAE</span>

Jiang, J., Xia, G. G., Carlton, D. B., Anderson, C. N., & Miyakawa, R. H. (2020, May). Transformer vae: A hierarchical model for structure-aware and interpretable music representation learning. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 516-520). IEEE.

<img src="images/Transformer VAE.png" width="200" height="200">

[Paper](https://ieeexplore.ieee.org/document/9054554)


### <span id="2019deep" style="color:#A8FF9E; font-size:20.0pt">2019</span>

#### <span id="tonicnet" style="color:#FF9EC3; font-size:15.0pt">TonicNet</span>

Peracha, O. (2019). Improving polyphonic music models with feature-rich encoding. arXiv preprint arXiv:1911.11775.

<img src="images/TonicNet.jpg" width="200" height="200">

[Paper](https://arxiv.org/abs/1911.11775)


#### <span id="lakhnes" style="color:#FF9EC3; font-size:15.0pt">LakhNES</span>

Donahue, C., Mao, H. H., Li, Y. E., Cottrell, G. W., & McAuley, J. (2019). LakhNES: Improving multi-instrumental music generation with cross-domain pre-training. arXiv preprint arXiv:1907.04868.

<img src="images/LakhNES.png" width="200" height="200">

[Paper](https://arxiv.org/abs/1907.04868)


#### <span id="r-transformer" style="color:#FF9EC3; font-size:15.0pt">R-Transformer</span>

Wang, Z., Ma, Y., Liu, Z., & Tang, J. (2019). R-transformer: Recurrent neural network enhanced transformer. arXiv preprint arXiv:1907.05572.

<img src="images/R-Transformer.png" width="400" height="200">

[Paper](https://arxiv.org/abs/1907.05572)


#### <span id="maia" style="color:#FF9EC3; font-size:15.0pt">Maia Music Generator</span>

<img src="images/Maia Music Generator.png" width="400" height="200">

[Web](https://maia.music.blog/2019/05/13/maia-a-new-music-generator/)


#### <span id="counterpoint-convolution" style="color:#FF9EC3; font-size:15.0pt">Coconet: Counterpoint by Convolution</span>

Huang, C. Z. A., Cooijmans, T., Roberts, A., Courville, A., & Eck, D. (2019). Counterpoint by convolution. arXiv preprint arXiv:1903.07227.

<img src="images/Coconet Counterpoint by Convolution.png" width="150" height="200">

[Paper](https://arxiv.org/abs/1903.07227) [Web](https://coconets.github.io/)


### <span id="2018deep" style="color:#A8FF9E; font-size:20.0pt">2018</span>

#### <span id="music-transformer" style="color:#FF9EC3; font-size:15.0pt">Music Transformer - Google Magenta</span>

Huang, C. Z. A., Vaswani, A., Uszkoreit, J., Shazeer, N., Simon, I., Hawthorne, et al. (2018). Music transformer. arXiv preprint arXiv:1809.04281.

<img src="images/Music Transformer.png" width="400" height="100">

[Web](https://magenta.tensorflow.org/music-transformer) [Poster](Images/transformer_poster.jpg) [Paper](https://arxiv.org/pdf/1809.04281.pdf)


#### <span id="imposing-structure" style="color:#FF9EC3; font-size:15.0pt">Imposing Higher-level Structure in Polyphonic Music</span>

Lattner, S., Grachten, M., & Widmer, G. (2018). Imposing higher-level structure in polyphonic music generation using convolutional restricted boltzmann machines and constraints. Journal of Creative Music Systems, 2, 1-31.

<img src="images/Imposing Higher-level Structure in Polyphonic Music.png" width="400" height="200">


[Paper](https://arxiv.org/pdf/1612.04742.pdf)

#### <span id="music-vae" style="color:#FF9EC3; font-size:15.0pt">MusicVAE - Google Magenta</span>

Roberts, A., Engel, J., Raffel, C., Hawthorne, C., & Eck, D. (2018, July). A hierarchical latent vector model for learning long-term structure in music. In International Conference on Machine Learning (pp. 4364-4373). PMLR.

<img src="images/MusicVAE.png" width="400" height="200">

[Web](https://magenta.tensorflow.org/music-vae) [Paper](https://arxiv.org/pdf/1803.05428.pdf) [Code](https://github.com/tensorflow/magenta/tree/master/magenta/models/music_vae) [Google Colab](https://colab.research.google.com/notebooks/magenta/music_vae/music_vae.ipynb) [Explanation](https://medium.com/@musicvaeubcse/musicvae-understanding-of-the-googles-work-for-interpolating-two-music-sequences-621dcbfa307c)


### <span id="2017deep" style="color:#A8FF9E; font-size:20.0pt">2017</span>

#### <span id="morpheus" style="color:#FF9EC3; font-size:15.0pt">MorpheuS</span>

Herremans, D., & Chew, E. (2017). MorpheuS: generating structured music with constrained patterns and tension. IEEE Transactions on Affective Computing, 10(4), 510-523.

<img src="images/MorpheuS.png" width="200" height="200">

[Paper](https://arxiv.org/pdf/1812.04832.pdf)

#### <span id="music-gan" style="color:#FF9EC3; font-size:15.0pt">Polyphonic GAN</span>

Lee, S. G., Hwang, U., Min, S., & Yoon, S. (2017). Polyphonic music generation with sequence generative adversarial networks. arXiv preprint arXiv:1710.11418.

<img src="images/Polyphonic GAN 1.png" width="350" height="150">

<img src="images/Polyphonic GAN 2.png" width="350" height="100">

[Paper](https://arxiv.org/abs/1710.11418)


#### <span id="bach-chorales-lstm" style="color:#FF9EC3; font-size:15.0pt">BachBot - Microsoft</span>

Liang, F. T., Gotham, M., Johnson, M., & Shotton, J. (2017, October). Automatic Stylistic Composition of Bach Chorales with Deep LSTM. In ISMIR (pp. 449-456).

<img src="images/BachBot 1.png" width="350" height="100">

<img src="images/BachBot 2.png" width="350" height="100">

[Paper](https://www.microsoft.com/en-us/research/publication/automatic-stylistic-composition-of-bach-chorales-with-deep-lstm/) [Liang Master Thesis 2016](https://www.mlmi.eng.cam.ac.uk/files/feynman_liang_8224771_assignsubmission_file_liangfeynmanthesis.pdf)


#### <span id="musegan" style="color:#FF9EC3; font-size:15.0pt">MuseGAN</span>

Dong, H. W., Hsiao, W. Y., Yang, L. C., & Yang, Y. H. (2018, April). Musegan: Multi-track sequential generative adversarial networks for symbolic music generation and accompaniment. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 32, No. 1).

<img src="images/MuseGAN.png" width="400" height="150">

[Web](https://salu133445.github.io/musegan/) [Paper](https://arxiv.org/pdf/1709.06298.pdf) [Poster](Images/musegan_ismir2017.jpg) [GitHub](https://github.com/salu133445/musegan)


#### <span id="lstm-composing" style="color:#FF9EC3; font-size:15.0pt">Composing Music with LSTM</span>

Johnson, D. D. (2017, April). Generating polyphonic music using tied parallel networks. In International conference on evolutionary and biologically inspired music and art (pp. 128-143). Springer, Cham.

<img src="images/Composing Music with LSTM.png" width="250" height="150">

[Paper](https://link.springer.com/chapter/10.1007/978-3-319-55750-2_9) [Web](https://www.danieldjohnson.com/2015/08/03/composing-music-with-recurrent-neural-networks/) [GitHub](https://github.com/danieldjohnson/biaxial-rnn-music-composition) [Blog](https://www.danieldjohnson.com/2015/08/03/composing-music-with-recurrent-neural-networks/)


#### <span id="organ" style="color:#FF9EC3; font-size:15.0pt">ORGAN</span>

Guimaraes, G. L., Sanchez-Lengeling, B., Outeiral, C., Farias, P. L. C., & Aspuru-Guzik, A. (2017). Objective-reinforced generative adversarial networks (ORGAN) for sequence generation models. arXiv preprint arXiv:1705.10843.

<img src="images/ORGAN.png" width="400" height="100">

[Paper](https://arxiv.org/abs/1705.10843)


#### <span id="midinet" style="color:#FF9EC3; font-size:15.0pt">MidiNet</span>

Yang, L. C., Chou, S. Y., & Yang, Y. H. (2017). MidiNet: A convolutional generative adversarial network for symbolic-domain music generation. arXiv preprint arXiv:1703.10847.

<img src="images/MidiNet.png" width="400" height="150">

[Paper](https://arxiv.org/abs/1703.10847)


### <span id="2016deep" style="color:#A8FF9E; font-size:20.0pt">2016</span>

#### <span id="deepbach" style="color:#FF9EC3; font-size:15.0pt">DeepBach</span>

Hadjeres, G., Pachet, F., & Nielsen, F. (2017, July). Deepbach: a steerable model for bach chorales generation. In International Conference on Machine Learning (pp. 1362-1371). PMLR.

<img src="images/DeepBach.png" width="200" height="250">

[Web](http://www.flow-machines.com/history/projects/deepbach-polyphonic-music-generation-bach-chorales/) [Paper](https://arxiv.org/pdf/1612.01010.pdf) [Code](https://github.com/Ghadjeres/DeepBach)


#### <span id="fine-tuning-rl" style="color:#FF9EC3; font-size:15.0pt">Fine-Tuning with RL</span>

Jaques, N., Gu, S., Turner, R. E., & Eck, D. (2016). Generating music by fine-tuning recurrent neural networks with reinforcement learning.

<img src="images/Fine-Tuning with RL.png" width="400" height="200">

[Paper](https://research.google/pubs/pub45871/)


#### <span id="c-rnn-gan" style="color:#FF9EC3; font-size:15.0pt">C-RNN-GAN</span>

Mogren, O. (2016). C-RNN-GAN: Continuous recurrent neural networks with adversarial training. arXiv preprint arXiv:1611.09904.

<img src="images/C-RNN-GAN.png" width="300" height="200">

[Paper](https://arxiv.org/abs/1611.09904)


#### <span id="seqgan" style="color:#FF9EC3; font-size:15.0pt">SeqGAN</span>

Yu, L., Zhang, W., Wang, J., & Yu, Y. (2017, February). Seqgan: Sequence generative adversarial nets with policy gradient. In Proceedings of the AAAI conference on artificial intelligence (Vol. 31, No. 1).

<img src="images/SeqGAN.png" width="400" height="150">

[Paper](https://arxiv.org/abs/1609.05473)


### <span id="2002deep" style="color:#A8FF9E; font-size:20.0pt">2002</span>

#### <span id="seqgan" style="color:#FF9EC3; font-size:15.0pt">Temporal Structure in Music</span>

Eck, D., & Schmidhuber, J. (2002, September). Finding temporal structure in music: Blues improvisation with LSTM recurrent networks. In Proceedings of the 12th IEEE workshop on neural networks for signal processing (pp. 747-756). IEEE.

[Paper](https://ieeexplore.ieee.org/document/1030094)


### <span id="1990deep" style="color:#A8FF9E; font-size:20.0pt">1980s - 1990s</span>

Mozer, M. C. (1994). Neural network music composition by prediction: Exploring the benefits of psychoacoustic constraints and multi-scale processing. Connection Science, 6(2-3), 247-280.

[Paper](https://www.tandfonline.com/doi/abs/10.1080/09540099408915726)

### <span id="books-reviews-deep" style="color:#A8FF9E; font-size:25.0pt">Books and Reviews</span>

### <span id="books-deep" style="color:#3C8CE8; font-size:20.0pt">Books</span>

* Briot, J. P., Hadjeres, G., & Pachet, F. (2020). Deep learning techniques for music generation (pp. 1-249). Springer.

### <span id="reviews-deep" style="color:#3C8CE8; font-size:20.0pt">Reviews</span>

* Hernandez-Olivan, C., & Beltran, J. R. (2021). Music composition with deep learning: A review. arXiv preprint arXiv:2108.12290.
[Paper](https://arxiv.org/abs/2108.12290)

* Ji, S., Luo, J., & Yang, X. (2020). A Comprehensive Survey on Deep Music Generation: Multi-level Representations, Algorithms, Evaluations, and Future Directions. arXiv preprint arXiv:2011.06801.
[Paper](https://arxiv.org/abs/2011.06801)

* Briot, J. P., Hadjeres, G., & Pachet, F. D. (2017). Deep learning techniques for music generation--a survey. arXiv preprint arXiv:1709.01620.
[Paper](https://arxiv.org/abs/1709.01620)


## <span id="deep-learning-audio-generation" style="color:#9EB1FF; font-size:25.0pt">4. Audio Generation</span>

### <span id="2023audiodeep" style="color:#A8FF9E; font-size:20.0pt">2023</span>

#### <span id="vall-e-x-music" style="color:#FF9EC3; font-size:15.0pt">Vall-E X</span>

Zhang, Z., Zhou, L., Wang, C., Chen, S., Wu, Y., Liu, S., ... & Wei, F. (2023). Speak Foreign Languages with Your Own Voice: Cross-Lingual Neural Codec Language Modeling. arXiv preprint arXiv:2303.03926.

[Paper](https://arxiv.org/abs/2303.03926)

#### <span id="ernie-music" style="color:#FF9EC3; font-size:15.0pt">ERNIE Music</span>

Zhu, P., Pang, C., Wang, S., Chai, Y., Sun, Y., Tian, H., & Wu, H. (2023). ERNIE-Music: Text-to-Waveform Music Generation with Diffusion Models. arXiv preprint arXiv:2302.04456.

[Paper](https://arxiv.org/abs/2302.04456)

#### <span id="multi-source-diffusion-models" style="color:#FF9EC3; font-size:15.0pt">Multi-Source Diffusion Models</span>

Mariani, G., Tallini, I., Postolache, E., Mancusi, M., Cosmo, L., & Rodolà, E. (2023). Multi-Source Diffusion Models for Simultaneous Music Generation and Separation. arXiv preprint arXiv:2302.02257.

[Paper](https://arxiv.org/abs/2302.02257) [Samples](https://gladia-research-group.github.io/multi-source-diffusion-models/)

#### <span id="singsong" style="color:#FF9EC3; font-size:15.0pt">SingSong</span>

Donahue, C., Caillon, A., Roberts, A., Manilow, E., Esling, P., Agostinelli, A., ... & Engel, J. (2023). SingSong: Generating musical accompaniments from singing. arXiv preprint arXiv:2301.12662.

[Paper](https://arxiv.org/abs/2301.12662) [Samples](https://storage.googleapis.com/sing-song/index.html)

#### <span id="audioldm" style="color:#FF9EC3; font-size:15.0pt">AudioLDM</span>

Liu, H., Chen, Z., Yuan, Y., Mei, X., Liu, X., Mandic, D., ... & Plumbley, M. D. (2023). AudioLDM: Text-to-Audio Generation with Latent Diffusion Models. arXiv preprint arXiv:2301.12503.

[Paper](https://arxiv.org/abs/2301.12503) [Samples](https://audioldm.github.io/) [GitHub] (https://github.com/haoheliu/AudioLDM)

#### <span id="mousai" style="color:#FF9EC3; font-size:15.0pt">Mousai</span>

Schneider, F., Jin, Z., & Schölkopf, B. (2023). Mo\^ usai: Text-to-Music Generation with Long-Context Latent Diffusion. arXiv preprint arXiv:2301.11757.

[Paper](https://arxiv.org/abs/2301.11757)

#### <span id="make-an-audio" style="color:#FF9EC3; font-size:15.0pt">Make-An-Audio</span>

Huang, R., Huang, J., Yang, D., Ren, Y., Liu, L., Li, M., ... & Zhao, Z. (2023). Make-An-Audio: Text-To-Audio Generation with Prompt-Enhanced Diffusion Models. arXiv preprint arXiv:2301.12661.

[Paper](https://arxiv.org/abs/2301.12661) [Samples](https://text-to-audio.github.io/)

#### <span id="noise2music" style="color:#FF9EC3; font-size:15.0pt">Noise2Music</span>

Huang, Q., Park, D. S., Wang, T., Denk, T. I., Ly, A., Chen, N., ... & Han, W. (2023). Noise2Music: Text-conditioned Music Generation with Diffusion Models. arXiv preprint arXiv:2302.03917.

[Paper](https://arxiv.org/abs/2302.03917) [Samples](https://google-research.github.io/noise2music/)

#### <span id="msanii" style="color:#FF9EC3; font-size:15.0pt">Msanii</span>

Maina, K. (2023). Msanii: High Fidelity Music Synthesis on a Shoestring Budget. arXiv preprint arXiv:2301.06468.

[Paper](https://arxiv.org/abs/2301.06468)

#### <span id="musiclm" style="color:#FF9EC3; font-size:15.0pt">MusicLM</span>

Agostinelli, A., Denk, T. I., Borsos, Z., Engel, J., Verzetti, M., Caillon, A., ... & Frank, C. (2023). Musiclm: Generating music from text. arXiv preprint arXiv:2301.11325.

[Paper](https://arxiv.org/abs/2301.11325) [Samples](https://google-research.github.io/seanet/musiclm/examples/) [Dataset](https://www.kaggle.com/datasets/googleai/musiccaps)


### <span id="2022audiodeep" style="color:#A8FF9E; font-size:20.0pt">2022</span>

#### <span id="musika" style="color:#FF9EC3; font-size:15.0pt">Musika</span>

Pasini, M., & Schlüter, J. (2022). Musika! Fast Infinite Waveform Music Generation. arXiv preprint arXiv:2208.08706.

[Paper](https://arxiv.org/abs/2208.08706)

#### <span id="audiolm" style="color:#FF9EC3; font-size:15.0pt">AudioLM</span>

Borsos, Z., Marinier, R., Vincent, D., Kharitonov, E., Pietquin, O., Sharifi, M., ... & Zeghidour, N. (2022). Audiolm: a language modeling approach to audio generation. arXiv preprint arXiv:2209.03143.

[Paper](https://arxiv.org/abs/2209.03143) [Samples](https://google-research.github.io/seanet/audiolm/examples/)

### <span id="2021audiodeep" style="color:#A8FF9E; font-size:20.0pt">2021</span>

#### <span id="rave" style="color:#FF9EC3; font-size:15.0pt">RAVE</span>

Caillon, A., & Esling, P. (2021). RAVE: A variational autoencoder for fast and high-quality neural audio synthesis. arXiv preprint arXiv:2111.05011.

[Paper](https://arxiv.org/abs/2111.05011) [GitHub](https://github.com/acids-ircam/RAVE)


### <span id="2020audiodeep" style="color:#A8FF9E; font-size:20.0pt">2020</span>

#### <span id="musenet" style="color:#FF9EC3; font-size:15.0pt">Jukebox - OpenAI</span>

<img src="images/Jukebox.png" width="400" height="150">

[Web](https://openai.com/blog/jukebox/) [Paper](https://arxiv.org/abs/2005.00341) [GitHub](https://github.com/openai/jukebox/)

### <span id="2017audiodeep" style="color:#A8FF9E; font-size:20.0pt">2017</span>

#### <span id="musenet" style="color:#FF9EC3; font-size:15.0pt">MuseNet - OpenAI</span>

[Web](https://openai.com/blog/musenet/)

## <span id="datasets" style="color:#9EB1FF; font-size:25.0pt">5. Datasets</span>

* JSB Chorales Dataset [Web](http://www-ens.iro.umontreal.ca/~boulanni/icml2012)

* Maestro Dataset [Web](https://magenta.tensorflow.org/datasets/maestro)

* The Lakh MIDI Dataset v0.1 [Web](https://colinraffel.com/projects/lmd/) [Tutorial IPython](https://nbviewer.jupyter.org/github/craffel/midi-dataset/blob/master/Tutorial.ipynb)

* MetaMIDI Dataset [Web](https://metacreation.net/metamidi-dataset/) [Zenodo](https://zenodo.org/record/5142664)


## <span id="journals" style="color:#9EB1FF; font-size:25.0pt">6. Journals and Conferences</span>

* International Society for Music Information Retrieval (ISMIR) [Web](https://www.ismir.net/)

* IEEE Signal Processing (ICASSP) [Web](https://signalprocessingsociety.org/publications-resources)

* ELSEVIER Signal Processing Journal [Web](https://www.journals.elsevier.com/signal-processing)

* Association for the Advancement of Artificial Intelligence (AAAI) [Web](https://www.aaai.org/)

* Journal of Artificial Intelligence Research (JAIR) [Web](https://www.jair.org/index.php/jair)

* International Joint Conferences on Artificial Intelligence (IJCAI) [Web](https://www.ijcai.org/)

* International Conference on Learning Representations (ICLR) [Web](https://iclr.cc)

* IET Signal Processing Journal [Web](https://digital-library.theiet.org/content/journals/iet-spr)

* Journal of New Music Research (JNMR) [Web](https://www.tandfonline.com/loi/nnmr20)

* Audio Engineering Society - Conference on Semantic Audio (AES) [Web](http://www.aes.org/)

* International Conference on Digital Audio Effects (DAFx) [Web](http://dafx.de/)


## <span id="authors" style="color:#9EB1FF; font-size:25.0pt">7. Authors</span>

* David Cope [Web](http://artsites.ucsc.edu/faculty/cope/)

* Colin Raffel [Web](https://colinraffel.com/)

* Jesse Engel [Web](https://jesseengel.github.io/)

* Douglas Eck [Web](http://www.iro.umontreal.ca/~eckdoug/)

* Anna Huang [Web](https://mila.quebec/en/person/anna-huang/)

* François Pachet [Web](https://www.francoispachet.fr/)

* Jeff Ens [Web](https://jeffens.com/)

* Philippe Pasquier [Web](https://www.sfu.ca/siat/people/research-faculty/philippe-pasquier.html)


## <span id="labs" style="color:#9EB1FF; font-size:25.0pt">8. Research Groups and Labs</span>

* Google Magenta [Web](https://magenta.tensorflow.org/)

* Audiolabs Erlangen [Web](https://www.audiolabs-erlangen.de/)

* Music Informatics Group [Web](https://musicinformatics.gatech.edu/)

* Music and Artificial Intelligence Lab [Web](https://musicai.citi.sinica.edu.tw/)

* Metacreation Lab [Web](https://metacreation.net/)


## <span id="apps" style="color:#9EB1FF; font-size:25.0pt">9. Apps for Music Generation with AI</span>

* AIVA (paid) [Web](https://www.aiva.ai/)

* Amper Music (paid) [Web](https://www.ampermusic.com/)

* Ecrett Music (paid) [Web](https://ecrettmusic.com/)

* Humtap (free, iOS) [Web](https://www.humtap.com/)

* Amadeus Code (free/paid, iOS) [Web](https://amadeuscode.com/top)

* Computoser (free) [Web](computoser.com)

* Brain.fm (paid) [Web](https://www.brain.fm/login?next=/app/player)


## <span id="other-resources" style="color:#9EB1FF; font-size:25.0pt">10. Other Resources</span>

* Bustena (web in spanish to learn harmony theory) [Web](http://www.bustena.com/curso-de-armonia-i/)


