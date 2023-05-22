from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r200"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 32
config.lr = 0.005
config.verbose = 2000
config.dali = False

config.rec = ('/home/pc0/projects/arcface/data/real_images_dataset/'
              'win500m_overlap150m_samples200_input224px')
config.num_classes = 2025
config.num_image = 2025 * 200
config.num_epoch = 10
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
config.save_all_states = True
