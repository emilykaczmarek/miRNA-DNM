Application of Deep Neural Maps (Pesteie et al., 2018) for microRNA-based cancer clustering and interpretation. Original code and paper can be found at https://github.com/mpslxz/DNM. The code has been modified for application to miRNA sequence data, the autoencoder architecture for 1D data (miRNA_AE) is updated and activation gradients are added (second cell of the DNM.ipynb file). This code has been converted to a notebook (.ipynb) from the original python file.
DNM-miRNA is an unsupervised representation learning-clustering application. With this method, high dimensional data can be reduced into lower dimensional latent features using an Autoencoder (AE), which are subsequently clustered using a Self-Organizing Map (SOM). Joint fine-tuning of the two methods tailors the latent space specifically for more accurate clustering. Following training, the SOM can be visualized and used for classification, interpretation of clusters, and analysis of individual samples/abnormal sample features through activation gradients.

<a href="url"><img src="https://user-images.githubusercontent.com/52331761/145657730-36d68701-50c6-491b-870e-2af8af3668da.png" height="525" width="875" ></a>


It is recommended to run DNM.ipynb in Google Colab. Otherwise, a full list of required packages and versions can be found in Google Colab by running the following command: !pip list –v. The code is built using Theano and the example 
This code can be downloaded and imported into a Google Colab notebook. Running the first cell sets up the environment for the code, and the second contains the DNM class. The third cell trains the DNM, and cells 4, 5, and 6 are used for interpretation after training through heatmaps, labelled maps, and activation gradients, respectively.

An example dataset is imported and used in the demo code from sklearn (breast cancer data), however any dataset can be imported into Colab and used with the DNM. Data should be formatted with samples (e.g., patients) as rows and features (e.g., miRNAs) as columns. 

The inputs for the DNM initialization are as follows:

input_image_size: feature dimensionality of input data<br/>
latent_size: size of smallest autoencoder layer (features reduced to this dimensionality)<br/>
lattice_size: dimensions of the SOM<br/>
ae_arch_class: autoencoder architecture to use (can design autoencoders for specific data types)<br/>
name: name of trial being run<br/>
init_params: weight initialization<br/>
sigma: parameter for Gaussian neighborhood function<br/>
alpha: parameter for weight updates <br/>
BATCH_SIZE: number of samples inputted before updating the network parameters<br/>
ae_lr: learning rate for training autoencoder<br/>
lmbd: parameter of the autoencoder update<br/>
som_pretrain_lr: learning rate for training self-organizing map<br/>
dnm_map_lr: learning rate for joint fine-tuning of autoencoder and self-organizing map<br/>

These variables can be tuned for specific applications to achieve optimal performance. The main tuning we recommend is the latent_size and lattice_size. To initialize the DNM, the following can be run: 

deep_map = DNM(input_image_size= len(X_train[0]),
                        latent_size= latent_size,
                        lattice_size= lattice_size,
                        ae_arch_class= miRNA_AE, 
                        name='miRNA_test',
                        init_params=None, 
                        sigma=None,
                        alpha=None,
                        BATCH_SIZE=16,
                        ae_lr=1e-3,
                        lmbd=1e-6,
                        som_pretrain_lr=0.005,
                        dnm_map_lr=0.05)


After initialization, training is implemented using: 

 deep_map.train(x_train=X_train.astype('float32'),  
                                dnm_epochs=1500, trial_name='test', name='test', location_name=None,
                                pre_train_epochs=[2500, 1500]) 


Following training, there are multiple ways of interpreting the DNM. First, a heatmap can be created to determine where the highest density of samples were mapped. This is an unsupervised, unlabeled way of determining clusters of data. First, locations of data should be acquired using the following command: 

locs = deep_map.get_locations(X_train.astype('float32'))
Next, the heatmap can be created using the function 

compute_scaled_kde_neoplastic(lattice_size, np.array(locs))

<a href="url"><img src="https://user-images.githubusercontent.com/52331761/145660097-b1885ba4-d453-42d5-b73e-143633586149.png" height="300" width="300" ></a>


If a subset of the data is labelled, this can be visualized using labelled_plot, where the following plot is outputted:
labelled_plot(lattice_size, locs, y_train, cols)

<a href="url"><img src="https://user-images.githubusercontent.com/52331761/145660108-cafc9b72-e04a-4564-b719-90278af55146.png" height="300" width="300" ></a>



Lastly, the most informative features found by the autoencoder of the DNM can be extracted using the function feat_vis(X_train)
Individual samples can be used as input to the function to find features corresponding to specific samples, or can be grouped to find features of classes/all data points. 

