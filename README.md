Currently I 've implemented the Rnn-rbm network described in the paper [Modeling Temporal Dependencies in High-Dimensional Sequences:Application to Polyphonic Music Generation and Transcription](http://arxiv.org/pdf/1206.6392v1.pdf). A very simple implementation can be found on [deeplearning.net](http://deeplearning.net/tutorial/rnnrbm.html)

I am trying to make it work better using [blocks](https://github.com/mila-udem/blocks) and make it more modular
- [x] Pretraining of Rbm
- [x] Pretraining of Rnn
- [ ] Variable layer number Rbm
- [ ] Modular Rnn block with any number/kind of rnns. [This would solve it](https://github.com/mila-udem/blocks/issues/46)

Unimportant stuff
- [ ] Get file of midis and train on them
- [ ] Somehow sample mp3 to create dataset
