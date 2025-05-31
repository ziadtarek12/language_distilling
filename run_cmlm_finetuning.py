# Training parameters
learning_rate = 5e-5
warmup_proportion = 0.1  # Using proportion instead of absolute steps
max_steps = 100000  # Full training uses 100k steps
num_steps_to_run = 10  # We'll do fewer steps for demonstration

# Optimizer using modern AdamW from transformers
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer
                if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(max_steps * warmup_proportion),
    num_training_steps=max_steps
)

# Training loop
running_loss = RunningMeter('loss')
model.train()

print("Starting CMLM fine-tuning...")
#Use a plain iterator instead of tqdm with len()
train_iter = iter(train_loader)
for step in range(num_steps_to_run):
    try:
        batch = next(train_iter)
    except StopIteration:
        # Restart iterator if we run out of batches
        train_iter = iter(train_loader)
        batch = next(train_iter)
        
    # Move batch to device
    batch = tuple(t.to(device) for t in batch)
    input_ids, input_mask, segment_ids, lm_label_ids = batch
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Create output mask from lm_label_ids for model forward pass
    output_mask = lm_label_ids != -1  # Masking for non-padded tokens
    
    # Forward pass with output_mask parameter
    loss = model(input_ids, segment_ids, input_mask, lm_label_ids, output_mask)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    running_loss(loss.item())
    print(f"Step {step}, Loss: {running_loss.val:.4f}")
    if step % 100 == 0:
        
        # Clear CUDA cache periodically to avoid memory issues
        torch.cuda.empty_cache()

# Save model checkpoint
torch.save(model.state_dict(), f"{output_dir}/model_step_{num_steps_to_run}.pt")
print(f"Model saved to {output_dir}/model_step_{num_steps_to_run}.pt")