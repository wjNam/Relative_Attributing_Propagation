# Interpreting Deep Neural Networks - Relative Attributing Propagation
Relative attributing propagation (RAP) decomposes the output predictions of DNNs with a new perspective of separating the relevant (positive) and irrelevant (negative) attributions according to the relative influence between the layers.
Detail description of this method is provided in our paper https://arxiv.org/pdf/1904.00605.pdf.

This paper has been accepted in AAAI 2020.

This code provides a implementation of RAP and LRP for Imagenet classification.
For implementing other explaining methods in the paper, we followed the tutorial of http://heatmapping.org and https://github.com/albermax/innvestigate.

![Alt text](/Figure.png)

# Requirements
	pytorch >= 1.2.0
	python >= 3.6
	matplotlib >= 1.3.1

# Run

	python main.py --method RAP --arc vgg
  	python main.py --method RAP --arc resnet
  



# Paper Citation
When using this code, please cite our paper.

	@misc{nam2019relative,
	title={Relative Attributing Propagation: Interpreting the Comparative Contributions of Individual Units in Deep Neural Networks},
	author={Woo-Jeoung Nam and Shir Gur and Jaesik Choi and Lior Wolf and Seong-Whan Lee},
	year={2019},
	eprint={1904.00605},
	archivePrefix={arXiv},
	primaryClass={cs.CV}
	}
