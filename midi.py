# Taken from https://github.com/EderSantana/fuel/tree/422dc3e354ef7d3723ccf06424614a4305e87ddf

import os
from collections import OrderedDict

import numpy as np
from fuel.utils import do_not_pickle_attributes
from fuel.datasets import IndexableDataset
from fuel import config


@do_not_pickle_attributes('indexables')
class MidiSequence(IndexableDataset):
    def __init__(self, which_dataset, which_set='train', **kwargs):
        '''Midi Datasets

        Parameters
        ---------

        which_datset: str
           one of the following 'jsb', 'midi', 'nottingham', 'muse'
        which_set: str
           which datset split 'train', 'valid' or 'test'

        ref: Boulanger-Lewandowski et. al. Modeling Temporal Dependencies
             in High-Dimensional Sequences: Application to Polyphonic
             Music Generation and Transcription.
             Download the dataset pickled datasets from here:
             http://www-etud.iro.umontreal.ca/~boulanni/icml2012
        '''
        self.which_set = which_set
        self.which_dataset = which_dataset

        self.sources = ('features', 'targets')

        super(MidiSequence, self).__init__(
            OrderedDict(zip(self.sources,
                            self._load_data(which_dataset, which_set))),
            **kwargs)

    def load(self):
        self.indexables = [data[self.start:self.stop] for source, data
                           in zip(self.provide_sources,
                                  self._load_data(
                                      self.which_dataset,
                                      self.which_set))
                           ]

    def _load_data(self, which_dataset, which_set):
        """
        which_dataset : choose between 'short' and 'long'
        """
        # Check which_set
        if which_set not in ['train', 'valid', 'test']:
            raise ValueError(which_set + " is not a recognized value. " +
                             "Valid values are ['train', 'valid', 'test'].")
        # Check which_dataset
        if not any([dset in which_dataset for dset in ['midi', 'nottingham', 'muse', 'jsb']]):
            raise ValueError(which_set + " is not a recognized value. " +
                             "Valid values are ['midi', 'nottingham', 'muse', 'jsb'].")
        _data_path = os.path.join(config.data_path, 'midi')
        _path = []
        if 'midi' in which_dataset:
            _path.append(os.path.join(_data_path, "Piano-midi.de.pickle"))
        if 'nottingham' in which_dataset:
            _path.append(os.path.join(_data_path, "Nottingham.pickle"))
        if 'muse' in which_dataset:
            _path.append(os.path.join(_data_path, "MuseData.pickle"))
        if 'jsb' in which_dataset:
            _path.append(os.path.join(_data_path, "JSBChorales.pickle"))
        data = []
        for p in _path:
            data += np.load(p)[which_set]


        features = np.asarray(
            [np.asarray(
                [self.list_to_nparray(time_step,
                                      88) for time_step in np.asarray(data[i][:-1])])
             for i in xrange(len(data))]
        )
        targets = np.asarray(
            [np.asarray([self.list_to_nparray(time_step,
                                              88) for time_step in np.asarray(data[i][1:])])
             for i in xrange(len(data))]
        )
        return features, targets

    def list_to_nparray(self, x, dim):
        y = np.zeros((dim,), dtype=np.float32)
        for i in x:
            y[i - 1 - 21] = 1
        return y

    def get_data(self, state=None, request=None):
        #batch = next(state)
        batch = super(MidiSequence, self).get_data(state, request)
        #if state is not None:
        #batch = [b.transpose(1,0,2) for b in state[request]]
        return batch


@do_not_pickle_attributes('indexables')
class MidiSequence2(IndexableDataset):
    def __init__(self, which_dataset, which_set='train', **kwargs):
        '''Midi Datasets

        Parameters
        ---------

        which_datset: str
           one of the following 'jsb', 'midi', 'nottingham', 'muse'
        which_set: str
           which datset split 'train', 'valid' or 'test'

        ref: Boulanger-Lewandowski et. al. Modeling Temporal Dependencies
             in High-Dimensional Sequences: Application to Polyphonic
             Music Generation and Transcription.
             Download the dataset pickled datasets from here:
             http://www-etud.iro.umontreal.ca/~boulanni/icml2012
        '''
        self.which_set = which_set
        self.which_dataset = which_dataset

        self.sources = ('features',)

        super(MidiSequence2, self).__init__(
            OrderedDict(zip(self.sources,
                            self._load_data(which_dataset, which_set))),
            **kwargs)

    def load(self):
        self.indexables = [data[self.start:self.stop] for source, data
                           in zip(self.provide_sources,
                                  self._load_data(
                                      self.which_dataset,
                                      self.which_set))
                           ]

    def _load_data(self, which_dataset, which_set):
        """
        which_dataset : choose between 'short' and 'long'
        """
        # Check which_set
        if which_set not in ['train', 'valid', 'test']:
            raise ValueError(which_set + " is not a recognized value. " +
                             "Valid values are ['train', 'valid', 'test'].")
        # Check which_dataset
        if not any([dset in which_dataset for dset in ['midi', 'nottingham', 'muse', 'jsb']]):
            raise ValueError(which_set + " is not a recognized value. " +
                             "Valid values are ['midi', 'nottingham', 'muse', 'jsb'].")
        _data_path = os.path.join(config.data_path, 'midi')
        _path = []
        if 'midi' in which_dataset:
            _path.append(os.path.join(_data_path, "Piano-midi.de.pickle"))
        if 'nottingham' in which_dataset:
            _path.append(os.path.join(_data_path, "Nottingham.pickle"))
        if 'muse' in which_dataset:
            _path.append(os.path.join(_data_path, "MuseData.pickle"))
        if 'jsb' in which_dataset:
            _path.append(os.path.join(_data_path, "JSBChorales.pickle"))
        data = []
        for p in _path:
            data += np.load(p)[which_set]

        features = np.asarray(
            [np.asarray(
                [self.list_to_nparray(time_step,
                                      88) for time_step in np.asarray(data[i])])
             for i in xrange(len(data))]
        )
        return features,

    def list_to_nparray(self, x, dim):
        y = np.zeros((dim,), dtype=np.float32)
        for i in x:
            y[i - 1 - 21] = 1
        return y

    def get_data(self, state=None, request=None):
        #batch = next(state)
        batch = super(MidiSequence2, self).get_data(state, request)
        #if state is not None:
        #batch = [b.transpose(1,0,2) for b in state[request]]
        return batch
