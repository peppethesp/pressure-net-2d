# Create and save dataset
import os
import shutil
from myPackageUtils.dataset_utils import save_dataset
# Import the needed user.defined functions
from importlib import reload
import n1_create_dataset.create_dataset as cd
reload(cd)

def execute(d_size, N_x, N_y, num_chunks):
    """
    - d_size: size of the single dataset file
    - N_x, N_y: dimension of the frames
    - num_chunks: number of pieces
    """
    dataset_folder = "saved_data/"
    filename = "dataset"
    # Delete the Dataset folder with the previous data
    overwrite = True
    path = (os.path.join(os.getcwd(), dataset_folder))
    if (overwrite == True) and (os.path.exists(path)):
        shutil.rmtree(path)

    for i in range(num_chunks):
        dataset = cd.create_dataset_spatial_filtering(d_size, N_x, N_y, filtering_radius=50)

        path = (os.path.join(os.getcwd(), (dataset_folder+filename+str(i))))
        save_dataset(dataset, path, True)

if __name__ == "__main__":
    execute(d_size=1000, N_x=128, N_y=128, num_chunks=10)