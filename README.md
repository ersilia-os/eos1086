# Text Embeddings Extraction using Pretrained Lamguage Models

Syntactic relationship and intrinsic information carried out in textual input data can be represented in the form of text embeddings. These embeddings can be utilised for the downstream tasks like classification, regression etc. BioMed-RoBERTa-base is a trandformer-based language model adapted from RoBERTa-base, pretrained on 2.68 million biomedical domain specific scientific papers (7.55B tokens and 47GB of data). The multi-layer structure of transformer captures different levels of representations in its different levels and learns a rich hierarchy of linguistic information. We are utilizing last hidden state of this hierachal structure to obtain these representations.

## Identifiers

* EOS model ID: `eos1086`
* Slug: `embeddings-extraction`

## Characteristics

* Input: `Text`
* Input Shape: `List`
* Task: `Representation`
* Output: `Descriptor`
* Output Type: `Float`
* Output Shape: `List`
* Interpretation: A list consisting of 768 float points values which is representation of textual input in numerical vector form.

## References

* [Publication](https://aclanthology.org/2020.acl-main.740/)
* [Source Code](https://huggingface.co/allenai/biomed_roberta_base)
* Ersilia contributor: [Femme-js](https://github.com/Femme-js)

## Ersilia model URLs
* [GitHub](https://github.com/ersilia-os/eos1086)

## Citation

If you use this model, please cite the [original authors](https://aclanthology.org/2020.acl-main.740/) of the model and the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff).

## License

This package is licensed under a GPL-3.0 license. The model contained within this package is licensed under a Apache-2.0 license.

Notice: Ersilia grants access to these models 'as is' provided by the original authors, please refer to the original code repository and/or publication if you use the model in your research.

## About Us

The [Ersilia Open Source Initiative](https://ersilia.io) is a Non Profit Organization ([1192266](https://register-of-charities.charitycommission.gov.uk/charity-search/-/charity-details/5170657/full-print)) with the mission is to equip labs, universities and clinics in LMIC with AI/ML tools for infectious disease research.

[Help us](https://www.ersilia.io/donate) achieve our mission!