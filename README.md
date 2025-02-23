## Setup

1. Create a venv/conda environment. This was previously tested with python 3.11.
2. Install required packages
   ```bash
   pip install -r requirements.txt
   ```

## Usage steps

1. Run data_collection.py -n {name of person} to collect face image data
2. Run train_mobilenetv3.py to train the model using data collected. Remember to set the size of classifier output to match the number of classes.
3. Test model using livestream_tf.py
4. Test attack using livestream_tf_attack.py
