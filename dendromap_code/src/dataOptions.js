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
		
    	dataset: "CIFAR-10 test",
    	model: "ResNet50 test",
    	cluster_filepath: "cifar10-test/clusters/cifar10-test.json",
        // class_cluster_filepath removed since we don't have class labels
    	image_filepath: "cifar10-test/images",
	},
    {
		
    	dataset: "Dumbo",
    	model: "ResNet50 Dumbo",
    	cluster_filepath: "dumbo/clusters/dumbo_clusters.json",
        // class_cluster_filepath removed since we don't have class labels
    	image_filepath: "dumbo/images",
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
