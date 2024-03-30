const options = [
	{
		dataset: "MNIST",
		model: "Autoencoder",
		cluster_filepath:
			"https://div-lab.github.io/dendromap-data/experimental/mnist/clusters/vae_mnist_clusters.json",
		classes_cluster_filepath: undefined,
		image_filepath:
			"https://div-lab.github.io/dendromap-data/experimental/mnist/images",
	},
	{
		
    	dataset: "CIFAR-10 Jin",
    	model: "ResNet50 Jin",
    	cluster_filepath: "cifar10/clusters/cifar10_test.json",
        // class_cluster_filepath removed since we don't have class labels
    	image_filepath: "cifar10/images",
	},
	{
		dataset: "CIFAR-100",
		model: "ResNet-50",
		cluster_filepath:
			"https://div-lab.github.io/dendromap-data/cifar100/clusters/cifar100_resnet50.json",
		class_cluster_filepath:
			"https://div-lab.github.io/dendromap-data/cifar100/clusters/cifar100_resnet50_classes.json",
		image_filepath:
			"https://div-lab.github.io/dendromap-data/cifar100/images",
	},
	// put your entry here and it will show up in the dropdown menu
];

export default options;
