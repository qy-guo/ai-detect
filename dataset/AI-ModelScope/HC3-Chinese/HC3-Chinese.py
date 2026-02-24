# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Human ChatGPT Comparison Corpus (HC3) Chinese Version"""

import json
import datasets

_CITATION = """\

"""

_DESCRIPTION = """\
Human ChatGPT Comparison Corpus (HC3) Chinese Version
"""

_HOMEPAGE = "https://github.com/Hello-SimpleAI/chatgpt-comparison-detection"

_LICENSE = ""

_BASE_URL = 'https://www.modelscope.cn/datasets/AI-ModelScope/HC3-Chinese/resolve/master/'

class NewDataset(datasets.GeneratorBasedBuilder):
    """Human ChatGPT Comparison Corpus (HC3) Chinese Version"""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="all", version=VERSION, description="concatenation of all other subsets"),
        datasets.BuilderConfig(name="baike", version=VERSION, description="baike"),
        datasets.BuilderConfig(name="open_qa", version=VERSION, description="open_questions"),
        datasets.BuilderConfig(name="nlpcc_dbqa", version=VERSION, description="nlpcc_dbqa"),
        datasets.BuilderConfig(name="finance", version=VERSION, description="finance"),
        datasets.BuilderConfig(name="medicine", version=VERSION, description="medicine"),
        datasets.BuilderConfig(name="law", version=VERSION, description="law"),
        datasets.BuilderConfig(name="psychology", version=VERSION, description="psychology"),
    ]

    DEFAULT_CONFIG_NAME = "all"

    def _info(self):
        if self.config.name == "all":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "human_answers": datasets.Sequence(datasets.Value("string")),
                    "chatgpt_answers": datasets.Sequence(datasets.Value("string")),
                    "source": datasets.Value("string")
                }
            )
        else:
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "human_answers": datasets.Sequence(datasets.Value("string")),
                    "chatgpt_answers": datasets.Sequence(datasets.Value("string")),
                }
            )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls = _BASE_URL + self.config.name + '.jsonl'
        data_path = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_path, "split": "train"},
            ),
        ]

    def _generate_examples(self, filepath, split):
        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                if self.config.name == "all":
                    yield key, {
                        "id": key,
                        "question": data["question"],
                        "human_answers": data["human_answers"],
                        "chatgpt_answers": data["chatgpt_answers"],
                        "source": data["source"]
                    }
                else:
                    yield key, {
                        "id": key,
                        "question": data["question"],
                        "human_answers": data["human_answers"],
                        "chatgpt_answers": data["chatgpt_answers"],
                    }
