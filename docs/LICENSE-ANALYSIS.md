# License Analysis: Legal Considerations for This Fork

## Overview

This fork is based on [MFFV](https://github.com/I-love-Meat-Fr/MFFV) (TIFS 2024 paper by Huang et al.) and inspired by the [FingerVeinSyn-5M](https://github.com/EvanWang98/FingerVeinSyn-5M) dataset (ACM MM 2025 by Wang et al.). This document analyzes the licensing and legal considerations for derivative research.

## Components and Their Licenses

| Component | Original License | This Fork's Approach |
|-----------|------------------|----------------------|
| FingerVeinSyn-5M dataset (images) | CC BY 4.0 | Cite + use under same terms |
| FingerVeinSyn-5M annotations (XML, masks) | CC BY 4.0 | Cite + use under same terms |
| MFFV code (FVIA, FVAB, FE, PT) | MIT (per repo LICENSE) | Preserve MIT |
| MFFV-N dataset | Signed agreement required | Pending approval, see `MFFV License Agreement.pdf` |
| **ENG-SDK-C++ vendor SDK** | **NOT INCLUDED** | Excluded for license clarity |
| FV-USM dataset | Academic use TOS | Cite + TOS compliance |

## ENG-SDK-C++: Why We Excluded It

The original workspace contained `ENG-SDK-C++/` with vendor SDK binaries (.dll, .msi, .docx) for a commercial finger vein sensor. Issues:

1. **No clear license**: Binary files lack LICENSE files
2. **Vendor proprietary**: Likely closed-source commercial SDK
3. **Distribution risk**: Re-distributing may violate vendor terms
4. **Reverse engineering risk**: Cannot study binary code

**Decision:** This fork does NOT include `ENG-SDK-C++/`. If you need sensor functionality, contact the vendor directly for licensing.

If you need to use the vendor sensor:
1. Contact the vendor directly for licensing
2. Apply for IRB approval if collecting new data
3. Use the SDK API only for research purposes (not redistribution)

## Citation Requirements

### For FingerVeinSyn-5M dataset:
```bibtex
@inproceedings{wang2025fingerveinsyn,
  title={FingerVeinSyn-5M: A Million-Scale Dataset and Benchmark 
         for Finger Vein Recognition},
  author={Wang, Yifan and Gui, Jie and Yu, Baosheng and Li, Qi and 
          Sun, Zhenan and Kannala, Juho and Zhao, Guoying},
  booktitle={Proceedings of the 33rd ACM International Conference 
             on Multimedia (ACM MM 25')},
  pages={13038--13045},
  year={2025}
}
```

### For FV-USM dataset:
```bibtex
@inproceedings{asaari2014fusion,
  title={Fusion of Finger Vein and Finger Print for Identity 
         Verification},
  author={Asaari, M. S. M. and Suandi, S. A. and Rosdi, B. A.},
  booktitle={Proceedings of the International Conference on 
             Biometrics and Forensics (IWBF)},
  pages={1--6},
  year={2014},
  organization={IEEE}
}
```

### For MFFV-N dataset (when using):
```bibtex
@article{huang2024mffv,
  title={MFFV: An Open Multi-Spectrum Finger Vein Database and 
         Benchmark for Recognition Under Various Environmental 
         Conditions},
  author={Huang, Junduan and others},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2024}
}
```

## Dual Licensing for This Fork

This fork uses **dual licensing**:
- **Code (`src/`, `scripts/`, derived from MFFV):** MIT License (preserved from original)
- **Documentation (`docs/`, `README.md`):** CC BY 4.0
- **New datasets we generate:** CC BY 4.0 (to match original)
- **Modifications to existing MFFV code:** MIT (compatible with original)

## Ethical Considerations

Finger vein is biometric data. While synthetic datasets avoid most privacy regulations, real datasets require:
- IRB approval for collection
- Informed consent for subjects
- Compliance with regional biometric laws:
  - EU: GDPR Article 9 (biometric = sensitive)
  - US: Illinois BIPA, Texas CUBI
  - VN: Decree 13/2023/ND-CP
  - CN: PIPL

## Contact for Clarification

- **Original FingerVeinSyn-5M authors:** 230239767@seu.edu.cn (Wang Yifan)
- **Original MFFV authors:** runrunjun@163.com (Junduan Huang)
- **This fork:** [add your contact info]

Last updated: 2026-08-07