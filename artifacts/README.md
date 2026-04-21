# Artifact textures

Some corruptions require real-world texture images that ship with this
repository:

## `fingerprint.jpg`
A single image of a fingerprint on a white background. The `fingerprint`
corruption thresholds this texture to extract the ridges, Gaussian-blurs
them, and alpha-blends them onto the bounding-box region — so the texture
should be a high-contrast black-on-white fingerprint scan.

Expected path: `artifacts/corruptions/fingerprint.jpg`

## `flares/`
A directory of lens-flare patches drawn from the **Flares7K** dataset
(Dai et al., ECCV 2022). At runtime one flare is chosen (randomly, or
deterministically if `--seed` is set), resized to the bounding box, and
composited onto the image with brightness-based alpha.

Expected path: `artifacts/corruptions/flares/*.{png,jpg,jpeg}`

Please refer to the Flares7K dataset's own license for downstream use of
these patches: <https://github.com/ykdai/Flare7K>

## Missing artifacts

If you see a `FileNotFoundError` when running `fingerprint` or
`lens_flare`, make sure these files are present. All other corruptions
(`camouflage`, `dust_scratches`, `focus_shift`, `illumination_variation`,
`occlusion`) run with no external assets.
