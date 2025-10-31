import torch
import math
import os
from time import time


def generate_epoch_checkpoints(args):
    max_exp = int(math.log2(args.n_epoch))
    num_checkpoints = max_exp * args.save_freq + 1
    times = torch.logspace(0, max_exp, num_checkpoints, base=2).int()
    times = times.tolist() + [args.n_epoch]
    return list(set(times))


def generate_step_checkpoints(args):
    max_steps = args.n_epoch * (args.train_size // args.batch_size + int((args.train_size % args.batch_size) > 0))
    max_exp = int(math.log2(max_steps))
    num_checkpoints = max_exp * args.save_freq + 1
    times = torch.logspace(0, max_exp, num_checkpoints, base=2).int()
    times = times.tolist() + [max_steps]
    return list(set(times))


def evaluate_model(step, args, ddpm, trainloader, testloader, eval_func, loss_ema=None, time_wall=None):
    ddpm.train()
    loss = 0.0
    num_points = 0
    for x, _ in trainloader: # Computes the training loss over the entire training set (can be expensive for large trainingsets)
        num_points += x.shape[0]
        x = x.to(args.device)
        loss += ddpm(x, args.n_trajectories).item() * len(x)
    loss /= num_points
    
    log_step = {}
    log_step["step"] = step
    log_step["loss"] = loss
    log_step["loss_ema"] = loss_ema
    log_step["Wall_time"] = time_wall

    ddpm.eval()
    with torch.no_grad():
        x_train   = next(iter(trainloader))[0].to(args.device)
        x_test    = next(iter(testloader))[0].to(args.device)
        x_samples = ddpm.sample(1024, (x.shape[1], x.shape[2]), args.device)

        log_step["Train_losses_per_time"] = eval_func['Train_losses_per_time'](ddpm, x_train)
        log_step["Test_losses_per_time"] = eval_func['Test_losses_per_time'](ddpm, x_test)
        log_step["Fraction_of_copies"] = eval_func['Fraction_of_copies'](x_samples)
        log_step["Valid_samples"] = eval_func['Valid_samples'](x_samples)

    # for key, val in log_step.items():
    #     print(f"{key} : {val}", flush=True)
    time_str = f"{log_step['Wall_time']:.0f}" if log_step['Wall_time'] is not None else "N/A"
    string = f"Step {log_step['step']}: train loss={log_step['loss']:.4f} - Valid={log_step['Valid_samples'][0]:.4f} - Copies={log_step['Fraction_of_copies']:.4f} - Wall time={time_str}s"
    print(string, flush=True)
    # print(f"Copies : {log_step['Fraction_of_copies']}", flush=True)
    # print(f"Valid : {log_step['Valid_samples'][0]}", flush=True)

    return log_step


def print_epoch(i, time0, loss_ema, loss):
    time_wall = time()
    print(f"Epoch {i}, wall t {(time_wall-time0):.0f}s : loss_ema={loss_ema:.4f}; loss={loss:.4f}", flush=True)




def train(trainloader, testloader, ddpm, optim_sched, args, eval_func={}):
    print(args, flush=True)

    log_results = {}
    epoch_checkpoints = generate_epoch_checkpoints(args)
    step_checkpoints = generate_step_checkpoints(args)
    optim, scheduler = optim_sched

    time0 = time()
    step = 0
    log_results[0] = evaluate_model(step, args, ddpm, trainloader, testloader, eval_func=eval_func)
    print_epoch(0, time0, log_results[0]["loss"], loss=log_results[0]["loss"])

    for i in range(1, args.n_epoch + 1):

        ddpm.train()
        loss_ema = None
        for i_batch, (x, _) in enumerate(trainloader):
            optim.zero_grad()
            x = x.to(args.device)
            loss = ddpm(x, args.n_trajectories)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            optim.step()
            scheduler.step()  # Step the learning rate scheduler
            step += 1
            current_epoch = (i-1) + (i_batch+1) / len(trainloader)

            if ((i_batch+1) == len(trainloader) and i in epoch_checkpoints) or (step in step_checkpoints):
                data = {}
                log_step = evaluate_model(
                    step, args, ddpm, trainloader, testloader, eval_func=eval_func, loss_ema=loss_ema, time_wall=time()-time0
                )
                log_results[current_epoch] = log_step

                # Create results directory if it doesn't exist
                os.makedirs("./results", exist_ok=True)
                
                # Save model
                torch.save(
                    ddpm.state_dict(), f"./results/{args.output}_ddpm_{args.dataset}.pt"
                )

                # Save logs
                torch.save(
                    {"results": log_results, "args": args},
                    f"./results/{args.output}_ddpm_{args.dataset}_logs.pt",
                )

                data["args"] = args
                data["results"] = log_results
                data["ddpm_state"] = ddpm.state_dict()

                yield data
            
        if i % args.print_period == 0:
            print_epoch(i, time0, loss_ema, loss=loss.item())
