# Ethics, privacy and limits of use

## Framing
A skull-based reconstruction supports the **identification of unknown deceased persons** (victims, unidentified remains). It produces an *approximation* of a face to generate investigative leads (e.g. a public appeal) that are then confirmed by DNA, dental records or fingerprints. It is not identification evidence on its own. If the paper mentions suspects, the realistic link is identifying a victim, which then helps an investigation; a reconstruction should never be used to point at a living suspect.

## Approvals
- Ethics committee / IRB approval or a documented exemption for secondary use of de-identified CT, at your institution.
- Follow every data use agreement (NMDID, TCIA). Most forbid redistribution and re-identification attempts.
- CQ500 is reported as non-commercial share-alike (CC BY-NC-SA 4.0); keep derived data under compatible terms.
- In Vietnam, check personal-data rules with your institution (Decree 13/2023/ND-CP on personal data protection and later legislation); facial geometry and health data are sensitive.

## Privacy in the paper and repository
- Never commit CT data, skin meshes or face renderings of real subjects to the repository (`.gitignore` excludes `data/`, `runs/`, `results/`).
- In figures, show error maps on a mean face or on subjects whose consent/DUA allows it; avoid photorealistic renderings of real people.
- Do not release trained weights if the DUA treats them as derived data; ask the data owners.

## Scientific limits to state
- Soft tissue of the nose tip, lips, ears and eyes is weakly determined by bone; expect the largest errors there.
- Body mass, age, sex and ancestry change tissue depth; report performance per group and avoid deploying a model on a population it was not trained on.
- Post-mortem changes (swelling, decomposition, supine posture, positioning aids) differ from living patients.
- Identification accuracy depends strongly on gallery size; small-gallery rank-1 numbers overstate real-world performance.
