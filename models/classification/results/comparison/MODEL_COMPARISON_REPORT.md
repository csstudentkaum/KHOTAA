# Comprehensive Model Comparison Report
## Diabetic Foot Ulcer Classification Using Deep Learning

**Report Generated:** 2026-02-10 17:35:48

---

## Executive Summary

This report presents a systematic comparison of 6 state-of-the-art deep learning architectures for automated diabetic foot ulcer (DFU) severity classification. The evaluation follows rigorous methodology suitable for peer-reviewed medical AI research.

### Models Evaluated

- **MobileNetV2** (MOBILENETV2)
- **ResNet50** (RESNET50)
- **ResNet101** (RESNET101)
- **EfficientNetV2S** (EFFICIENTNETV2S)
- **GoogLeNet** (GOOGLENET)
- **DenseNet121** (DENSENET)

### Dataset Overview

- **Total Samples:** 10062
- **Training Samples:** 9921
- **Validation Samples:** Held-out validation set (used in this analysis)
- **Test Samples:** 141 (RESERVED for final evaluation)
- **Number of Classes:** 4
- **Class Labels:** Grade 1, Grade 2, Grade 3, Grade 4
- **Validation Strategy:** 5-Fold Stratified Cross-Validation

### Class Distribution

- Grade 1: 2326 samples
- Grade 2: 2423 samples
- Grade 3: 2784 samples
- Grade 4: 2388 samples

---

## Methodology

### Evaluation Protocol

**Cross-Validation Strategy:**
- 5-fold stratified cross-validation maintaining class proportions
- Each model trained on 4 folds, validated on 1 fold (repeated 5 times)
- Performance aggregated across all folds to estimate robustness

**Performance Metrics:**
- **Accuracy:** Overall classification accuracy
- **Precision:** Positive predictive value (class-specific)
- **Recall/Sensitivity:** True positive rate (clinical relevance)
- **F1-Score:** Harmonic mean of precision and recall
- **Specificity:** True negative rate
- **AUC-ROC:** Area under receiver operating characteristic curve

**Statistical Analysis:**
- Paired t-tests for pairwise model comparisons
- Significance levels: p < 0.05 (*), p < 0.01 (**), p < 0.001 (***)
- Effect sizes reported as Cohen's d

---

## Results

### Top Performing Models

#### 🥇 1st Place: DenseNet121

- **Cross-Validation Accuracy:** 0.98% ± 0.00%
- **Macro F1-Score:** 0.9935
- **Macro AUC-ROC:** 0.9999
- **Sensitivity:** 0.9935
- **Specificity:** 0.9978
- **Training Epochs:** 29.8

#### 🥈 2nd Place: EfficientNetV2S

- **Cross-Validation Accuracy:** 0.98% ± 0.00%
- **Macro F1-Score:** 0.9890
- **Macro AUC-ROC:** 0.9990
- **Sensitivity:** 0.9893
- **Specificity:** 0.9963
- **Training Epochs:** 29.6

#### 🥉 3rd Place: MobileNetV2

- **Cross-Validation Accuracy:** 0.98% ± 0.00%
- **Macro F1-Score:** 0.9910
- **Macro AUC-ROC:** 0.9998
- **Sensitivity:** 0.9910
- **Specificity:** 0.9970
- **Training Epochs:** 27.8

### Key Findings

1. **Best Overall Performance:** DenseNet121 achieved the highest mean cross-validation accuracy of 0.98%

2. **Most Consistent Model:** DenseNet121 demonstrated the lowest standard deviation (0.00%), indicating reliable performance across different data splits

3. **Most Efficient Training:** MobileNetV2 converged fastest with an average of 27.8 epochs

4. **Best Discriminative Ability:** DenseNet121 achieved the highest macro AUC-ROC of 0.9999

5. **Best Sensitivity:** DenseNet121 achieved sensitivity of 0.9935, minimizing false negatives (critical for severe case detection)

---

## Clinical Recommendations

### For Clinical Deployment

**Primary Recommendation:** DenseNet121
- Rationale: Highest overall accuracy with strong performance across all metrics
- Use case: General DFU severity assessment in well-resourced clinical settings

**Alternative for Resource-Constrained Settings:** MobileNetV2
- Rationale: Fastest training convergence, potentially lower computational requirements
- Use case: Mobile health applications, point-of-care devices

**For Maximum Reliability:** DenseNet121
- Rationale: Most consistent performance across patient populations
- Use case: Multi-site clinical trials, diverse patient cohorts

### Clinical Interpretation Guidelines

- **High Sensitivity Required:** Use DenseNet121 to minimize missed severe cases
- **Balanced Performance:** Consider top 3 models for ensemble approaches
- **External Validation:** All models require validation on institution-specific data before clinical deployment

---

## Study Limitations

1. **Dataset Constraints:** Single-source dataset may not represent global DFU heterogeneity
2. **Validation Strategy:** Test set reserved; external validation pending
3. **Clinical Applicability:** Real-world performance and clinical utility not yet assessed
4. **Technical Considerations:** Optimal hyperparameters may vary by architecture
5. **Reporting Constraints:** Macro-averaged metrics may obscure class-specific variations

For detailed limitations, see Section 8 of the notebook.

---

## Generated Outputs

### Tables (CSV Format)
- `01_comprehensive_metrics.csv` - Complete metrics for all models
- `02_cv_performance.csv` - Cross-validation performance summary
- `03_per_class_metrics.csv` - Detailed per-class performance
- `04_model_ranking.csv` - Model ranking by performance

### Figures (Publication Quality)
- `fig1_accuracy_comparison.png` - Accuracy with confidence intervals
- `fig2_fold_distribution.png` - Distribution across CV folds
- `fig3_f1_heatmap.png` - Per-class F1-score heatmap
- `fig4_per_class_metrics.png` - Multi-metric per-class comparison
- `fig5_confusion_matrices.png` - Normalized confusion matrices
- `fig6_efficiency_analysis.png` - Training efficiency analysis
- `fig7_radar_chart.png` - Top 3 models radar chart

---

## Conclusion

This systematic comparison provides robust evidence for model selection in DFU classification tasks. The top-performing models demonstrate clinically relevant accuracy levels, though external validation and prospective clinical evaluation remain necessary before deployment.

**Next Steps:**
1. External validation on independent datasets
2. Prospective clinical trial
3. Integration with clinical decision support systems
4. Regulatory evaluation for medical device certification

---

*Report generated automatically from comprehensive validation metrics*  
*Analysis performed in accordance with medical AI research best practices*
