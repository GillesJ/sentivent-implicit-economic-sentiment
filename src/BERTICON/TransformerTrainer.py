from math import ceil
import logging

import pandas as pd
import torch
import torch.distributed as dist
from torch.nn import MSELoss, CrossEntropyLoss, Softmax, BCEWithLogitsLoss, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import numpy as np

try:
    from apex import amp
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

from Config import Config
from DfDataset import DfDataset
from Performance import Performance

#logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class TransformerTrainer:
    """ Trainer that executes the training, validation, and testing loops
        given all required information such as model, tokenizer, optimizer and so on. """
    def __init__(self,
                 model,
                 optimizer=None,
                 scheduler=None,
                 map_location=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.map_location = map_location

        self.config = Config.get()
        self.mopts = self.config.model
        self.popts = self.config.process
        self.topts = self.config.training

        self.device = self.popts['device']
        self.distributed = self.popts['distributed']
        self.is_first = self.popts['is_first']

        self.batch_size = self.topts['batch_size']
        self.fp16 = self.topts['fp16']
        self.num_labels = self.topts['num_labels']
        self.multi_label = self.topts['multi_label']
        self.task = self.topts['task']
        

        # When the model is wrapped as DistributedDataParallel,
        # its properties are not overtly available. Use .module to access them
        try:
            self.tokenizer = model.tokenizer_wrapper
        except AttributeError:
            self.tokenizer = model.module.tokenizer_wrapper

        self._set_data_loaders()
        if self.task == 'regression':
            self.criterion = MSELoss().to(self.device)
        else:
            if self.topts['label_weights']:
                self.criterion = CrossEntropyLoss(weight=torch.tensor(self.topts['label_weights'])).to(self.device)
            else:
                if self.multi_label == 0:
                    self.criterion = CrossEntropyLoss().to(self.device)
                else:
                    self.criterion = BCEWithLogitsLoss().to(self.device)

        #self.softmax = Softmax(dim=1).to(self.device) if self.num_labels > 1 else None
        self.softmax = Softmax(dim=1).to(self.device) if self.task == 'classification' else None
        self.sigmoid = Sigmoid().to(self.device)
        self.performer = Performance()

    def _prepare_lines(self, data, labels=False):
        """ Basic line preparation, strips away new lines.
            Can also prepare labels as the expected tensor type"""
        if labels:
            # For regression, cast to float (FloatTensor)
            # For classification, cast to int (LongTensor)
            if self.task == 'regression':
                if self.num_labels == 1:
                    out = torch.FloatTensor([float(item.rstrip()) for item in data])
                else:
                    out = []
                    for item in data:
                        label_ml = [float(i) for i in item.rstrip().split(',')]
                        out.append(label_ml)
                    out = torch.from_numpy(np.asarray(out)).float()
            else:
                if self.multi_label == 0:
                    out = torch.LongTensor([int(item.rstrip()) for item in data])
                else:
                    out = []
                    for item in data:
                        label_ml = [int(i) for i in item.rstrip().split(',')]
                        out.append(label_ml)
                    out = torch.from_numpy(np.asarray(out)).float()
        else:
            out = [item.rstrip() for item in data]

        return out

    def _process(self, do, epoch=0, inference=False):
        """ Runs the training, validation, or testing (for one epoch) """
        if do == 'train' and not inference:
            self.model.train()
            torch.set_grad_enabled(True)
        else:
            self.model.eval()
            torch.set_grad_enabled(False)

        if self.distributed:
            self.samplers[do].set_epoch(epoch)

        # only run tqdm progressbar for first process
        progress_bar = None
        if self.is_first:
            nro_batches = ceil(len(self.datasets[do]) / self.dataloaders[do].batch_size)
            desc = f"Epoch {epoch:,} ({do})" if do in {'train', 'valid'} else 'Test'
            progress_bar = tqdm(desc=desc, total=nro_batches)

        # Main loop: iterate over dataloader
        for batch_idx, data in enumerate(self.dataloaders[do], 1):
            # 0. Clear gradients
            if do == 'train' and not inference:
                self.performer.start_train_loop_time()
                self.optimizer.zero_grad()

            # 1. Data prep
            text = self._prepare_lines(data['text'])
            encoded_inputs = self.tokenizer.encode_batch_plus(text,
                                                              batch_pair=None,
                                                              pad_to_batch_length=True,
                                                              return_tensors='pt')

            if inference:
                sentence_ids = data['id'].to(self.device)
            else:
                if self.task == 'regression':
                    if self.num_labels == 1:
                        labels = data['label'].to(self.device, dtype=torch.int64)
                    else:
                        labels = self._prepare_lines(data['label'], labels=True)
                        labels = labels.to(self.device, dtype=torch.float64)
                else:
                    if self.multi_label == 0:
                        labels = data['label'].to(self.device, dtype=torch.int64)
                    else:
                        labels = self._prepare_lines(data['label'], labels=True)
                        labels = labels.to(self.device, dtype=torch.float64)
                

            encoded_inputs['input_ids'] = encoded_inputs['input_ids'].to(self.device)
            encoded_inputs['attention_mask'] = encoded_inputs['attention_mask'].to(self.device)
            encoded_inputs['token_type_ids'] = encoded_inputs['token_type_ids'].to(self.device)

            # 2. Predictions
            try:
                preds = self.model(**encoded_inputs)
            except RuntimeError as e:
                with open('error.log', 'w', encoding='utf-8') as fhout:
                    fhout.write(str(data) + '\n')
                    fhout.write(str(encoded_inputs) + '\n')

                with open('trace.log', 'w', encoding='utf-8') as traceout:
                    traceout.write(str(e) + '\n')

                raise RuntimeError()

            if self.task == 'regression':
                preds = preds.squeeze()
                if not inference:
                    if self.num_labels == 1:
                        loss = self.criterion(preds.view(-1), labels.view(-1)).unsqueeze(0)
                    else:
                        #losses = []
                        #for i in range(self.num_labels):
                        #    preds_i = preds[:, i]
                        #    labels_i = labels[:, i]
                        #    loss_i = self.criterion(preds_i, labels_i)
                        #    losses.append(loss_i)
                        #losses = torch.tensor(losses, requires_grad=True)
                        #loss = torch.mean(losses).unsqueeze(0).to(self.device)
                        loss = self.criterion(preds.to(self.device, dtype=torch.float64), labels.to(self.device, dtype=torch.float64)).unsqueeze(0)
                        #print(loss)
                        #print(loss.type())


            else:
                if not inference:
                    loss = self.criterion(preds, labels).unsqueeze(0)
                if self.multi_label == 0:
                    probs = self.softmax(preds)
                    preds = torch.topk(probs, 1).indices.squeeze()
                else:
                    preds = self.sigmoid(preds)
                    probs = preds
                    preds = preds > 0.5
                    preds = preds.squeeze()

            # 3. Optimise during training
            if do == 'train' and not inference:
                if self.fp16:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                self.optimizer.step()

            # 4. Save results
            if inference:
                self.performer.update('sentence_ids', do, sentence_ids)
                if self.task == 'classification':
                    self.performer.update('probs', do, probs)
            else:
                self.performer.update('labels', do, labels)
                self.performer.update('losses', do, loss)
            self.performer.update('preds', do, preds)

            if progress_bar:
                upd_step = min(self.popts['world_size'], progress_bar.total - progress_bar.n)
                progress_bar.update(upd_step)

            if do == 'train' and not inference:
                self.performer.end_train_loop_time()

        if progress_bar:
            progress_bar.close()

    def _save_model(self, valid_metric):
        """ Saves current model as well as additional information. """
        info_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'valid_loss': self.performer.min_valid_loss,
            'valid_metric': valid_metric,
            'epoch': self.performer.last_saved_epoch,
            'training_time': self.performer.training_time
        }

        if self.fp16 and AMP_AVAILABLE:
            info_dict['amp_state_dict'] = amp.state_dict()

        torch.save(info_dict, 'tmp-checkpoint.pth')

    def _set_data_loaders(self):
        """ Create datasets and their respective dataloaders.
            See DfDataset.py """
        train_file = self.topts['files']['train'] if 'train' in self.topts['files'] else None
        valid_file = self.topts['files']['valid'] if 'valid' in self.topts['files'] else None
        test_file = self.topts['files']['test'] if 'test' in self.topts['files'] else None

        self.datasets = {
            'train': DfDataset(train_file, sep=self.topts['sep']) if train_file is not None else None,
            'valid': DfDataset(valid_file, sep=self.topts['sep']) if valid_file is not None else None,
            'test': DfDataset(test_file, sep=self.topts['sep']) if test_file is not None else None
        }

        if train_file:
            logger.info(f"Training set size: {len(self.datasets['train'])}")
        if valid_file:
            logger.info(f"Validation set size: {len(self.datasets['valid'])}")
        if test_file:
            logger.info(f"Test set size: {len(self.datasets['test'])}")

        self.samplers = {part: None for part in ('train', 'valid', 'test')}
        if self.distributed:
            self.samplers = {
                'train': DistributedSampler(self.datasets['train']) if train_file is not None else None,
                'valid': DistributedSampler(self.datasets['valid']) if valid_file is not None else None,
                'test': DistributedSampler(self.datasets['test']) if test_file is not None else None
            }

        self.dataloaders = {
            # no advantage of running more than 1 num_workers here
            'train': DataLoader(self.datasets['train'], batch_size=self.batch_size, shuffle=False, num_workers=1,
                                pin_memory=True, sampler=self.samplers['train'] if self.samplers['train'] else None)
                     if train_file is not None else None,
            'valid': DataLoader(self.datasets['valid'], batch_size=self.batch_size, shuffle=False, num_workers=1,
                                pin_memory=True, sampler=self.samplers['valid'] if self.samplers['valid'] else None)
                     if valid_file is not None else None,
            'test': DataLoader(self.datasets['test'], batch_size=self.batch_size, shuffle=False, num_workers=1,
                               pin_memory=True, sampler=self.samplers['test'] if self.samplers['test'] else None)
                    if test_file is not None else None
        }

    def load_model(self, checkpoint_f, eval_mode=False):
        """ Load checkpoint, especially used for testing. """
        checkpoint = torch.load(checkpoint_f, map_location=self.map_location if self.map_location else self.device)

        # running in DDP mode will add module., so might need to remove that in the keys
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError:
            # maybe doing this in comprehension is too memory intensive?
            checkpoint['model_state_dict'] = {k.replace('module.', ''): v for k, v in
                                              checkpoint['model_state_dict'].items()}
            self.model.load_state_dict(checkpoint['model_state_dict'])

        # only load optimizer if not in eval mode
        if not eval_mode and self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if self.fp16 and AMP_AVAILABLE and 'amp_state_dict' in checkpoint:
                amp.load_state_dict(checkpoint['amp_state_dict'])

        # If we don't do this, it might lead to CUDA OOM issues for larger models
        # I'm not sure why since after exiting the function, I'd expect the variable to
        # be free-able, but this seems to work after testing.
        del checkpoint

    def infer(self, checkpoint_f):
        """ Predicts values for the file in 'test' """
        self.load_model(checkpoint_f, eval_mode=True)

        self._process('test', inference=True)
        self.performer.gather_cat('test')
        #print(self.performer.gathered['test'])
        if self.task == 'regression':
            if self.num_labels == 1:
                pred_type = float
                data = {
                'id': self.performer.gathered['test']['sentence_ids'].numpy().astype(int),
                'pred': self.performer.gathered['test']['preds'].numpy().astype(pred_type)
                }
            else:
                pred = self.performer.gathered['test']['preds'].numpy().tolist()
                new_pred = []
                for label in pred:
                    label = [float(instance) for instance in label]
                    label = [str(instance) for instance in label]
                    label = ','.join(label)
                    new_pred.append(label)
                #print(new_pred)
                data = {
                'id': self.performer.gathered['test']['sentence_ids'].numpy().astype(int),
                'pred': new_pred
                }

        else:
            if self.topts['multi_label'] == 0:
                pred_type = int
                prob = self.performer.gathered['test']['probs'].numpy().tolist()
                new_prob = []
                for label in prob:
                    label = [float(instance) for instance in label]
                    label = [str(instance) for instance in label]
                    label = ','.join(label)
                    new_prob.append(label)
                data = {
                'id': self.performer.gathered['test']['sentence_ids'].numpy().astype(int),
                'pred': self.performer.gathered['test']['preds'].numpy().astype(pred_type),
                'prob': new_prob
                }
            else:
                pred = self.performer.gathered['test']['preds'].numpy().tolist()
                prob = self.performer.gathered['test']['probs'].numpy().tolist()
                new_pred = []
                for label in pred:
                    label = [int(instance) for instance in label]
                    label = [str(instance) for instance in label]
                    label = ','.join(label)
                    new_pred.append(label)
                new_prob = []
                for label in prob:
                    label = [float(instance) for instance in label]
                    label = [str(instance) for instance in label]
                    label = ','.join(label)
                    new_prob.append(label)
                #print(new_pred)
                data = {
                'id': self.performer.gathered['test']['sentence_ids'].numpy().astype(int),
                'pred': new_pred,
                'prob': new_prob
                }
        

        df = pd.DataFrame.from_dict(data)
        #df.to_csv('multilabel_data/output/predictions/predictions.csv', index=False)
        #filename = 'eventdna/predictions_' + str(self.topts['files']['test'][36:])
        filename = self.topts['pred_path']
        df.to_csv(filename, index=False)

    def test(self, checkpoint_f):
        """ Wraps testing a given model. Actual testing is done in `self._process()`. """
        self.load_model(checkpoint_f, eval_mode=True)

        self._process('test')
        self.performer.gather_cat('test')
        avg_test_loss, test_2nd_metric, report = self.performer.evaluate_test()

        return avg_test_loss, test_2nd_metric, report

    def train(self):
        """ Entry point to start training the model. Will run the outer epoch loop containing
            training and validation. Also implements early stopping, set by `self.patience`.
            Actual training/validating is done in `self._process()` """
        logger.info('Training started.')

        done_training = False
        fig = None
        #best_valid_loss = 0
        #best_valid_2nd_metric = 0
        best_train_loss = 0
        best_train_2nd_metric = 0
        for epoch in range(1, self.topts['epochs'] + 1):
            # TRAINING
            self._process('train', epoch)

            self.performer.gather_cat('train')

            avg_train_loss = torch.empty(1).to(self.device)
            if self.is_first:
                avg_train_loss, train_2nd_metric, fig, save_model, done_training = self.performer.evaluate(epoch)
                if save_model:
                    self._save_model(train_2nd_metric)
                    best_train_loss = avg_train_loss.item()
                    best_train_2nd_metric = train_2nd_metric

            if self.distributed:
                # broadcast done_training: due to a bug, broadcasting booleans does not work as expected
                # to by-pass this, cast to long (int) first, and then recast as bool
                # see https://github.com/pytorch/pytorch/issues/24137
                done_training = torch.tensor(done_training).long().to(self.device)
                torch.distributed.broadcast(done_training, src=0)
                done_training = bool(done_training.item())

                # broadcast avg_valid_loss for the scheduler
                torch.distributed.broadcast(avg_valid_loss, src=0)

            if done_training:
                break

            # adjust learning rate with scheduler
            if self.scheduler is not None:
                self.scheduler.step(avg_train_loss)

            """
            # VALIDATION
            self._process('valid', epoch)
            self.performer.gather_cat('train', 'valid')

            avg_valid_loss = torch.empty(1).to(self.device)
            if self.is_first:
                avg_valid_loss, valid_2nd_metric, fig, save_model, done_training = self.performer.evaluate(epoch)
                if save_model:
                    self._save_model(valid_2nd_metric)
                    best_valid_loss = avg_valid_loss.item()
                    best_valid_2nd_metric = valid_2nd_metric

            if self.distributed:
                # broadcast done_training: due to a bug, broadcasting booleans does not work as expected
                # to by-pass this, cast to long (int) first, and then recast as bool
                # see https://github.com/pytorch/pytorch/issues/24137
                done_training = torch.tensor(done_training).long().to(self.device)
                torch.distributed.broadcast(done_training, src=0)
                done_training = bool(done_training.item())

                # broadcast avg_valid_loss for the scheduler
                torch.distributed.broadcast(avg_valid_loss, src=0)

            if done_training:
                break

            # adjust learning rate with scheduler
            if self.scheduler is not None:
                self.scheduler.step(avg_valid_loss)
            """

            # clear the performer's saved labels, losses, preds
            self.performer.clear()

        #return fig, best_valid_loss, best_valid_2nd_metric
        return fig, best_train_loss, best_train_2nd_metric
