# EncCluster: Scalable Functional Encryption in FL with Weight Clustering and Probabilistic Filters

Federated Learning (FL) enables model training across decentralized devices by communicating solely local model updates to an aggregation server. Although such limited data sharing makes FL more secure than centralized approached, FL remains vulnerable to inference attacks during model update transmissions. Existing secure aggregation approaches rely on differential privacy or cryptographic schemes like Functional Encryption (FE) to safeguard individual client data. However, such strategies can reduce performance or introduce unacceptable computational and communication overheads on clients running on edge devices with limited resources. In this work, we present *EncCluster*, a novel method that integrates model compression through weight clustering with recent decentralized FE and privacy-enhancing data encoding using probabilistic filters to deliver strong privacy guarantees in FL without affecting model performance or adding unnecessary burdens to clients. We performed a comprehensive evaluation, spanning various datasets and architectures, to demonstrate *EncCluster*'s scalability across encryption levels. Our findings reveal that *EncCluster* significantly reduces communication costs — below even conventional FedAvg — and accelerates encryption by more than four times over all baselines; at the same time, it maintains high model accuracy and enhanced privacy assurances.

A complete description of our work can be found in our paper [arXiv](TODO).

```
git clone https://github.com/FederatedML/EncCluster
cd EncCluster/src
```

## Dependencies
Create a new Python enviroment (virtualenvs, anacoda, etc.) and install all required packages via:
```console
foo@bar:~$ pip install -r requirements.txt
```

## Executing experiments
From the `src` (root) directory of this repo, run:

```console
# FedLN-NNC
foo@bar:~$ TODO
```
You can configure all federated parameters (i.e. number of federated rounds, number of clients, type of probablistic filter, dataset, model, etc.,) by passing them as command line arguments.

# Reference
If you use this repository, please consider citing:

<pre>
  TODO
</pre>
