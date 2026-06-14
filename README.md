# DIAGNOSE: Distillation for Impartial And Group-aware Neural Output Standardization and Equity

![DIAGNOSE](assets/framework.png)  

Paper Abstract: Medical imaging models often exhibit systematic performance disparities across demographic groups, limiting equitable clinical deployment. We propose DIAGNOSE, a fairness-aware multi-teacher knowledge distillation framework that decouples cohort-specific learning from unified model optimization. Our approach operates in three stages: (i) fairness-aware backbone pretraining using adaptive sample weighting, (ii) cohort-specific teacher heads capturing group-tailored decision boundaries, and (iii) unified student distillation via a hybrid loss combining multi-teacher knowledge transfer with fairness-weighted supervision. Crucially, sensitive attributes are required only during training; the final model performs inference without demographic information. We evaluate DIAGNOSE on seven medical imaging benchmarks spanning dermatology, chest radiography, ophthalmology, and neuroimaging across multiple sensitive attributes (age, gender, race, skin type). DIAGNOSE achieves the highest average overall AUROC (88.5%) and worst-group AUROC (85.3%), while substantially reducing fairness disparities attaining 38-69% lower Equalized Odds Difference and the lowest Demographic Parity Difference (12.3%) among evaluated methods. While AUROC gap is not minimized in every setting, our results demonstrate strong utility-equity trade-offs, establishing DIAGNOSE as an effective approach for learning equitable medical image classifiers.

## Environment & Hardware Setup
This codebase uses PyTorch 2.0+ features (`torch.compile`) and automatic mixed precision (AMP).

**Key dependencies:**
- Python 3.8+
- PyTorch 2.0+
- timm
- transformers
- fairlearn

**Quickstart:**
```bash
# Clone the repository (replace with actual URL)
git clone https://github.com/your-username/diagnose.git
cd diagnose

# Install dependencies
pip install -r requirements.txt
```

After installing dependencies, update `config.py` to set the data root and select the dataset. The key fields are `Config.data_path`, `Config.dataset_config['name']`, and `Config.dataset_config['sensitive_attr']`.

### Datasets
Due to the data use agreements, we cannot directly share the download link. Please register and download datasets using the links from the table below:

| **Dataset**  | **Access**                                                                                    |
|--------------|-----------------------------------------------------------------------------------------------|
| CheXpert     | https://stanfordmlgroup.github.io/competitions/chexpert/                                      |
| OL3I         | https://stanfordaimi.azurewebsites.net/datasets/3263e34a-252e-460f-8f63-d585a9bfecfc          |
| PAPILA       | https://www.nature.com/articles/s41597-022-01388-1#Sec6                                       |
| HAM10000     | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T               |
| Oasis-1      | https://www.oasis-brains.org/#data                                                            |
| Fitzpatrick17k | https://github.com/mattgroh/fitzpatrick17k                                                  |
| Harvard-GF3300  |  https://ophai.hms.harvard.edu/datasets/harvard-glaucoma-fairness-3300-samples/            |

## Citation
If you find this code or our methodology useful for your research, please cite our paper:

```bibtex
@article{your_paper_placeholder,
  title={DIAGNOSE: Distillation for Impartial And Group-aware Neural Output Standardization and Equity},
  author={First Author and Co-authors},
  journal={Placeholder Journal},
  year={202X}
}
```
