import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import subprocess
from torch.nn.utils.rnn import pack_padded_sequence
import argparse
from torch.optim.lr_scheduler import StepLR
import datetime
from models_Encoder import CNN_Encoder
from models_Decoder import *
from datasets import *
from eval_MModalCC import evaluate_transformer
import sys
sys.path.append(r"C:\Users\TUBITAK\Desktop\RSICC_v2\SECONDCCpap")
from utils import *

seed = 1
torch.manual_seed(seed)

metrics_list = []
losses_output = []
AVG_losses_output = []
top5_accuracy_output = []
batch_time_output = []

train_model_sonuc_map = {}
text_terminal = " "

rogue_l_output = []
cider_output = []
bleu_4_output = []
rogue_l_nochange_output = []
cider_nochange_output = []
bleu_4_nochange_output = []
meteor1_nochange_output = []
meteor1_change_output = []
meteor1_output = []
rogue_l_change_output = []
cider_change_output = []
bleu_4_change_output = []

val_model_sonuc_map = {}

def print_with_json(text):
    global text_terminal
    print(text)
    text_terminal += str(text) + "\n"

def train(args, train_loader, encoder_image,encoder_image2,encoder_feat, decoder, criterion, encoder_image_optimizer,encoder_image_optimizer2,encoder_image_lr_scheduler,encoder_image_lr_scheduler2,encoder_feat_optimizer,encoder_feat_lr_scheduler, decoder_optimizer, decoder_lr_scheduler, epoch):

    encoder_image.train()
    encoder_image2.train()
    encoder_feat.train()
    decoder.train()



    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs_our = AverageMeter()
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    best_score = 0.  # BLEU-4 score right now

    for i, (img_pairs, caps, caplens) in enumerate(train_loader):

        data_time.update(time.time() - start)
        img_pairs = img_pairs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        imgs_A = img_pairs[:, 0, :, :, :]
        imgs_B = img_pairs[:, 1, :, :, :]
        sem_A = img_pairs[:, 2, :, :, :]
        sem_B = img_pairs[:, 3, :, :, :]

        imgs_A = encoder_image(imgs_A) # imgs_A: [batch_size,1024, 14, 14]
        imgs_B = encoder_image(imgs_B) # batch time = 0.35
        sem_A = encoder_image2(sem_A)
        sem_B = encoder_image2(sem_B) #batch time  0.4

        fused_feat, fused_feat2 = encoder_feat(imgs_A, sem_A, imgs_B, sem_B) # encoder_out: (S, batch, feature_dim) # fused_feat: (S, batch, feature_dim) # buyuk tensor atama yavaslatior (#batch time = 0.5)
        scores, caps_sorted, decode_lengths, sort_ind = decoder(fused_feat,fused_feat2, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        loss = criterion(scores, targets)

        # Back prop.
        decoder_optimizer.zero_grad()
        encoder_feat_optimizer.zero_grad()
        if encoder_image_optimizer is not None:
            encoder_image_optimizer.zero_grad()
        if encoder_image_optimizer2 is not None:
            encoder_image_optimizer2.zero_grad()
        loss.backward()

        # Clip gradients
        if args.grad_clip is not None:
            clip_gradient(decoder_optimizer, args.grad_clip)
            if encoder_image_optimizer is not None:
                clip_gradient(encoder_image_optimizer, args.grad_clip)
            if encoder_image_optimizer2 is not None:
                clip_gradient(encoder_image_optimizer2, args.grad_clip)

        # Update weights
        decoder_optimizer.step()
        decoder_lr_scheduler.step()
        encoder_feat_optimizer.step()
        encoder_feat_lr_scheduler.step()
        if encoder_image_optimizer is not None:
            encoder_image_optimizer.step()
            encoder_image_lr_scheduler.step()
            encoder_image_optimizer2.step()
            encoder_image_lr_scheduler2.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 1)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()
        if i % args.print_freq == 0:
            print_with_json("Epoch: {}/{} step: {}/{} Loss: {} AVG_Loss: {} Top-5 Accuracy: {} Batch_time: {}s".format(epoch+0, args.epochs, i+0, len(train_loader), losses.val, losses.avg, top5accs.val, batch_time.val))
            losses_output.append(losses.val)
            AVG_losses_output.append(losses.avg)
            top5_accuracy_output.append(top5accs.val)
            batch_time_output.append(batch_time.val)


def main(args, meteor_output=None):
    print_with_json(args)
    global metrics_list
    print_with_json(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))

    start_epoch = 0
    best_score = 0.  # BLEU-4 score right now
    epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

    # Read word map
    word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Encoder
    encoder_image = CNN_Encoder(NetType=args.encoder_image, method=args.decoder)
    encoder_image2 = CNN_Encoder(NetType=args.encoder_image, method=args.decoder)

    encoder_image.fine_tune(args.fine_tune_encoder)
    encoder_image2.fine_tune(args.fine_tune_encoder)

    # set the encoder_dim
    encoder_image_dim = 1024 #resnet101

    if args.encoder_feat == 'MCCFormers_diff_as_Q':
        encoder_feat = MCCFormers_diff_as_Q(feature_dim=encoder_image_dim, dropout=0.5, h=14, w=14, d_model=512, n_head=args.n_heads,
                               n_layers=args.n_layers)

    # Decoder
    if args.decoder == 'trans':
        decoder = DecoderTransformer(feature_dim=args.feature_dim_de,
                                      vocab_size=len(word_map),
                                      n_head=args.n_heads,
                                      n_layers=args.decoder_n_layers,
                                      dropout=args.dropout)


    encoder_image_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder_image.parameters()),
                                         lr=args.encoder_lr) if args.fine_tune_encoder else None
    encoder_image_optimizer2 = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder_image2.parameters()),
                                               lr=args.encoder_lr) if args.fine_tune_encoder else None

    if args.checkpoint is not 'None':
        filename = os.listdir(args.checkpoint)
        checkpoint_path = os.path.join(args.checkpoint, filename[0])
        checkpoint = torch.load(checkpoint_path, map_location=str(device))

    encoder_image_lr_scheduler = StepLR(encoder_image_optimizer, step_size=1316, gamma=1) if args.fine_tune_encoder else None
    encoder_image_lr_scheduler2 = StepLR(encoder_image_optimizer2, step_size=1316,
                                        gamma=1) if args.fine_tune_encoder else None

    encoder_feat_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder_feat.parameters()),
                                         lr=args.encoder_lr)
    encoder_feat_lr_scheduler = StepLR(encoder_feat_optimizer, step_size=1316, gamma=1)

    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=args.decoder_lr)
    decoder_lr_scheduler = StepLR(decoder_optimizer,step_size=1316,gamma=1)

    # Move to GPU, if available
    encoder_image = encoder_image.to(device)
    encoder_image2 = encoder_image2.to(device)
    encoder_feat = encoder_feat.to(device)
    decoder = decoder.to(device)



    print_with_json("Checkpoint_savepath:{}".format(args.savepath))
    print_with_json("Encoder_image_mode:{}   Encoder_feat_mode:{}   Decoder_mode:{}".format(args.encoder_image,args.encoder_feat,args.decoder))
    print_with_json("encoder_layers {} decoder_layers {} n_heads {} dropout {} encoder_lr {} "
          "decoder_lr {}".format(args.n_layers, args.decoder_n_layers, args.n_heads, args.dropout,
                                         args.encoder_lr, args.decoder_lr))

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
    # If your data elements are a custom type, or your collate_fn returns a batch that is a custom type.
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_folder, args.data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, args.epochs):

        # Decay learning rate if there is no improvement for x consecutive epochs, and terminate training after x
        if epochs_since_improvement == args.stop_criteria:
            print_with_json("the model has not improved in the last {} epochs".format(args.stop_criteria))
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 3 == 0:
            adjust_learning_rate(decoder_optimizer, 0.7)
            if args.fine_tune_encoder and encoder_image_optimizer is not None:
                print_with_json(encoder_image_optimizer)
                # adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        print_with_json(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
        train(args,
              train_loader=train_loader,
              encoder_image=encoder_image,
              encoder_image2=encoder_image2,
              encoder_feat=encoder_feat,
              decoder=decoder,
              criterion=criterion,
              encoder_image_optimizer=encoder_image_optimizer,
              encoder_image_optimizer2= encoder_image_optimizer2,
              encoder_image_lr_scheduler=encoder_image_lr_scheduler,
              encoder_image_lr_scheduler2 = encoder_image_lr_scheduler2,
              encoder_feat_optimizer=encoder_feat_optimizer,
              encoder_feat_lr_scheduler=encoder_feat_lr_scheduler,
              decoder_optimizer=decoder_optimizer,
              decoder_lr_scheduler=decoder_lr_scheduler,
              epoch=epoch)

        # One epoch's validation
        args.Split = "VAL"
        metrics, nochange_metrics,change_metrics = evaluate_transformer(args,
                            encoder_image=encoder_image,
                            encoder_image2=encoder_image2,
                            encoder_feat=encoder_feat,
                           decoder=decoder)
        args2 = args
        args2.Split = "TEST"
        metricss, nochange_metricss,change_metricss = evaluate_transformer(args2,
                            encoder_image=encoder_image,
                            encoder_image2=encoder_image2,
                            encoder_feat=encoder_feat,
                           decoder=decoder)


        metrics_list.append(metrics)
        recent_bleu4 = metrics["Bleu_4"]
        recent_CIDER = metrics["CIDEr"]
        bleu_4_output.append([metrics["Bleu_1"],metrics["Bleu_2"],metrics["Bleu_3"],metrics["Bleu_4"]])
        rogue_l_output.append(metrics["ROUGE_L"])
        meteor1_output.append(metrics["METEOR"])
        cider_output.append(metrics["CIDEr"])
        bleu_4_nochange_output.append([nochange_metrics["Bleu_1"],nochange_metrics["Bleu_2"],nochange_metrics["Bleu_3"],nochange_metrics["Bleu_4"]])
        rogue_l_nochange_output.append(nochange_metrics["ROUGE_L"])
        cider_nochange_output.append(nochange_metrics["CIDEr"])
        meteor1_nochange_output.append(nochange_metrics["METEOR"])
        bleu_4_change_output.append([change_metrics["Bleu_1"],change_metrics["Bleu_2"],change_metrics["Bleu_3"],
                                     change_metrics["Bleu_4"]])
        rogue_l_change_output.append(change_metrics["ROUGE_L"])
        cider_change_output.append(change_metrics["CIDEr"])
        meteor1_change_output.append(change_metrics["METEOR"])

        recentScore = recent_bleu4 #+ recent_CIDER
        # Check if there was an improvement
        is_best = recentScore > best_score
        best_score = max(recentScore, best_score)
        if not is_best:
            epochs_since_improvement += 1
            print_with_json("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        checkpoint_name = args.encoder_image + '_'+args.encoder_feat + '_' + args.decoder #_tengxun_aggregation
        save_checkpoint(args, checkpoint_name, epoch, epochs_since_improvement, encoder_image, encoder_image2,encoder_feat, decoder,
                        encoder_image_optimizer,encoder_image_optimizer2,encoder_feat_optimizer,decoder_optimizer, metrics, is_best)

    train_model_sonuc_map["losses"] = losses_output
    train_model_sonuc_map["avg_losses"] = AVG_losses_output
    train_model_sonuc_map["top5_acc"] = top5_accuracy_output
    val_model_sonuc_map["rogue_l"] = rogue_l_output
    val_model_sonuc_map["cider"] = cider_output
    val_model_sonuc_map["bleu_4"] = bleu_4_output
    val_model_sonuc_map["meteor"] = meteor1_output
    val_model_sonuc_map["rogue_l_nochange"] = rogue_l_nochange_output
    val_model_sonuc_map["cider_nochange"] = cider_nochange_output
    val_model_sonuc_map["meteor_nochange"] = meteor1_nochange_output
    val_model_sonuc_map["bleu_4_nochange"] = bleu_4_nochange_output
    val_model_sonuc_map["rogue_l_change"] = rogue_l_change_output
    val_model_sonuc_map["cider_change"] = cider_change_output
    val_model_sonuc_map["bleu_4_change"] = bleu_4_change_output
    val_model_sonuc_map["meteor_change"] = meteor1_change_output

    train_model_sonuc_json = json.dumps(train_model_sonuc_map,indent=4)
    val_model_sonuc_json = json.dumps(val_model_sonuc_map,indent=4)
    # Get the current date in the format YYYY-MM-DD
    current_date = datetime.date.today().strftime("%Y%m%d")

    # Define your save path
    output_save_path = args.savepath.replace('/model_dir', '')

    # Construct the filename with the current date
    file_name = f'{output_save_path}/train_{current_date}.json'
    file_name2 = f'{output_save_path}/val_{current_date}.json'
    file_name3 = f'{output_save_path}/terminal_text_{current_date}.txt'

    # Assuming you already have train_model_sonuc_json
    # Write the JSON data to the file
    with open(file_name3, 'w') as dosya:
        dosya.write(text_terminal)
    with open(file_name, 'w') as dosya:
        dosya.write(train_model_sonuc_json)
    with open(file_name2, 'w') as dosya:
        dosya.write(val_model_sonuc_json)

current_date = datetime.date.today().strftime("%Y%m%d")

if __name__ == '__main__':
    print_with_json("SECOND-CC")
    dosya_index = 0
    folder_path = f'./model_results/{current_date}_MModalCC_{dosya_index}'
    # while os.path.exists(folder_path):
    #     # If it doesn't exist, create it
    #     print(f"Folder '{folder_path}' already exists.")
    #     dosya_index += 1
    #     folder_path = f'./model_results/{current_date}_MModalCC_{dosya_index}'

    folder_path = f'./model_results/{current_date}_MModalCC_2'
    folder_path += '/model_dir'
    # os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created successfully.")

    parser = argparse.ArgumentParser(description='Image_Change_Captioning')
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


    # Data parameters
    parser.add_argument('--data_folder', default=r".\createdFileBlackAUG",
                        help='folder with data files saved by create_input_files.py.')
    parser.add_argument('--data_name', default="SECOND_CC_5_cap_per_img_10_min_word_freq",
                        help='base name shared by data files.')

    # Model parameters
    parser.add_argument('--encoder_image', default="resnet101", help='which model does encoder use?')
    parser.add_argument('--encoder_feat', default='MCCFormers_diff_as_Q')
    parser.add_argument('--decoder', default='trans')
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--feature_dim_de', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--stop_criteria', type=int, default=10, help='training stop if epochs_since_improvement == stop_criteria')
    parser.add_argument('--batch_size', type=int, default=28, help='batch_size')
    parser.add_argument('--print_freq', type=int, default=100, help='print training/validation stats every __ batches.')
    parser.add_argument('--workers', type=int, default=0, help='for data-loading; right now, only 0 works with h5pys in windows.')
    parser.add_argument('--encoder_lr', type=float, default= 5e-5, help='learning rate for encoder if fine-tuning.')# en son 5e-5 yap
    parser.add_argument('--decoder_lr', type=float, default= 5e-5, help='learning rate for decoder.')# en son 5e-5 yap
    parser.add_argument('--grad_clip', type=float, default=5., help='clip gradients at an absolute value of.')
    parser.add_argument('--fine_tune_encoder', type=bool, default=True, help='whether fine-tune encoder or not')

    parser.add_argument('--checkpoint', default="None",
                        help='path to checkpoint, None if none.')
    # Validation
    parser.add_argument('--Split', default="VAL", help='which')
    parser.add_argument('--beam_size', type=int, default=1, help='beam_size.')
    parser.add_argument('--savepath', default=folder_path)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = parser.parse_args()
    main(args)

    subprocess.run(f"python eval_MModalCC.py --data_folder {args.data_folder} --path {folder_path} --beam_size {args.beam_size} --data_name {args.data_name}")