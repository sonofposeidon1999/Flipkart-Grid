
import numpy as np

import uisrnn


SAVED_MODEL_NAME = 'pretrained/saved_model.uisrnn_benchmark'


def diarization_experiment(model_args, training_args, inference_args):


  predicted_labels = []
  test_record = []

  train_data = np.load('./ghostvlad/training_data.npz')
  train_sequence = train_data['train_sequence']
  train_cluster_id = train_data['train_cluster_id']
  train_sequence_list = [seq.astype(float)+0.00001 for seq in train_sequence]
  train_cluster_id_list = [np.array(cid).astype(str) for cid in train_cluster_id]

  model = uisrnn.UISRNN(model_args)

  # training
  model.fit(train_sequence_list, train_cluster_id_list, training_args)
  model.save(SAVED_MODEL_NAME)


def main():
  """The main function."""
  model_args, training_args, inference_args = uisrnn.parse_arguments()
  model_args.observation_dim = 512
  model_args.rnn_depth = 1
  model_args.rnn_hidden_size = 512
  training_args.enforce_cluster_id_uniqueness = False
  training_args.batch_size = 30
  training_args.learning_rate = 1e-4
  training_args.train_iteration = 3000
  training_args.num_permutations = 20
  # training_args.grad_max_norm = 5.0
  training_args.learning_rate_half_life = 1000
  diarization_experiment(model_args, training_args, inference_args)


if __name__ == '__main__':
  main()
