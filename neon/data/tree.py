# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from math import ceil
from neon import NervanaObject


class TreeDataset(NervanaObject):

    def __init__(self, dataset, batch_size, embeddings):
        assert self.be.bsz == 1, "Tree datasets do not currently support parallel batching, please set self.be.bsz to 1"
        # TODO: this batch size will always be 1

        # Have to explicitly set the batch size here because no current support for parallel batching, so have to do it
        # a bit differently

        self.shape = (len(dataset[0][0][0])*embeddings.shape[0], len(dataset[0][0]))
        self.embeddings = embeddings
        self.nbatches = ceil(len(dataset) / batch_size)
        self.ndata = len(dataset)

        self.batches = []
        self.batches.append([])
        batchno = 0
        for i, data in enumerate(dataset):

            self.batches[batchno].append(data)

            if((i+1)%batch_size == 0 and self.ndata > i+1):
                batchno += 1
                self.batches.append([])

    def __iter__(self):

        self.batch_index = 0
        while self.batch_index < self.nbatches:

            yield self.batches[self.batch_index][0]

            self.batch_index += 1
