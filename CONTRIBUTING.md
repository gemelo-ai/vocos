# Contributing to Vocos

Thanks for your interest in improving Vocos. Bug reports, documentation updates, compatibility fixes, and other
focused contributions are welcome.

## Reporting an issue

Please search the existing issues first. When reporting a bug, include a minimal reproduction, the expected and
observed behavior, relevant logs, and your Python, PyTorch, and platform versions. For audio-quality reports, describe
the input format, sampling rate, checkpoint, and inference path used.

## Proposing a change

Small fixes and documentation improvements can go directly to a pull request. For larger behavioral or architectural
changes, please open an issue first so the approach can be discussed before substantial implementation work begins.

## Pull requests

Before opening a pull request:

- Search the existing issues and pull requests to avoid duplicating work.
- Keep the change focused and explain the motivation and expected behavior.
- Preserve compatibility with the documented pretrained models whenever possible.
- Describe how you checked the change, including the Python and PyTorch versions used when relevant.

Pull requests should summarize the change, link any related issue, and call out compatibility implications. Tests or
clear validation steps are especially helpful for changes that affect model loading, tensor shapes, audio sampling
rates, or training behavior.

## Documentation

Documentation-only pull requests are welcome. Please keep examples concise, preserve the existing checkpoint names,
and verify that commands and links render correctly on GitHub.
