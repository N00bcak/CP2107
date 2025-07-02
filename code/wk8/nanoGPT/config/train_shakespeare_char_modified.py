# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-shakespeare-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 32 # Suppress batch size since our GPU is a baby, and we would like noisier gradients
block_size = 256 # context of up to 256 previous characters

# Longer, thinner model for greater expressiveness
n_layer = 8
n_head = 8
n_embd = 256
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 20000
lr_decay_iters = 20000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

dtype = 'bfloat16' # Use BF16 since our GPU supports it.
# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
