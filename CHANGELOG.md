# Changelog

## [0.1.1](https://github.com/S1M0N38/llm-fingerprint/compare/v0.1.0...v0.1.1) (2025-03-20)


### Bug Fixes

* **upload:** move initialization from `__init__` to main ([cfbac6b](https://github.com/S1M0N38/llm-fingerprint/commit/cfbac6b5cc59132bd762110bc5e1402a5d1189c7))
* **upload:** set centroid metadata key for samples to false ([721ef93](https://github.com/S1M0N38/llm-fingerprint/commit/721ef9335708adb2fe46ebb3f4892d65591b02f7))


### Documentation

* **CONTRIBUTING:** add how to contribute and project structure sections ([40bf332](https://github.com/S1M0N38/llm-fingerprint/commit/40bf3327a4ac492ca88ef4a8e8af6fccd53914c2))

## 0.1.0 (2025-03-19)


### Features

* choose sentences transformers device ([331f701](https://github.com/S1M0N38/llm-fingerprint/commit/331f701d83c27a538b0d8c3cddad7f4e912dbe6c))
* **cli:** accept multiple models in generate cmd ([81341b8](https://github.com/S1M0N38/llm-fingerprint/commit/81341b8d4a5edc403f59baa99dc840c89e2d3b4d))
* **cli:** add cli module ([efe7416](https://github.com/S1M0N38/llm-fingerprint/commit/efe74169d6a054b914a91cf6eee0114cfe27f527))
* **cli:** implement query command ([0e7fac5](https://github.com/S1M0N38/llm-fingerprint/commit/0e7fac5b3ea4cffa2b6db191a1ebf7b5902ad437))
* **cli:** implement upload_cmd ([9ace2b0](https://github.com/S1M0N38/llm-fingerprint/commit/9ace2b0b57d7b48f1d999d0579e9e01dba329794))
* **config:** add llama-cpp.yaml config ([9303216](https://github.com/S1M0N38/llm-fingerprint/commit/930321691c1688cddc084e89d334574075389193))
* **config:** add phi-4 and smolLM config for llama-cpp ([546366d](https://github.com/S1M0N38/llm-fingerprint/commit/546366dcb48d5a58ffed650953c4bb70b8a4d0a6))
* **config:** add suggested generation params in llama-cpp config ([512c806](https://github.com/S1M0N38/llm-fingerprint/commit/512c8065cf3abce4e3e089b876b9188b3b7c76d1))
* **generate:** add module for sample generation ([c2f1581](https://github.com/S1M0N38/llm-fingerprint/commit/c2f15818d965387a1dec6d9777fcf6a224816de8))
* **just:** add chroma-run and chroma-stop ([fc3f471](https://github.com/S1M0N38/llm-fingerprint/commit/fc3f4717c0cc192295c4075618348b0686bb6748))
* **just:** add model families as variable for generate ([92e6b15](https://github.com/S1M0N38/llm-fingerprint/commit/92e6b155ef8b18a1ddd0fd7fdc9602f8495d3820))
* **justfile:** add phi-4 and smolLM models ([5ca96ba](https://github.com/S1M0N38/llm-fingerprint/commit/5ca96ba47e2ce7add1977ad5645740284e62814f))
* **justfile:** add recipe for generate samples with a single model ([f534bf1](https://github.com/S1M0N38/llm-fingerprint/commit/f534bf158fd43c6d78298374800d0ffd677ca7d6))
* **justfile:** add recipe for sample generation for local models ([6951db9](https://github.com/S1M0N38/llm-fingerprint/commit/6951db9e190d4da007466d01df8d8e02ba61d94c))
* **models:** add models for llm_fingerprint ([2965df7](https://github.com/S1M0N38/llm-fingerprint/commit/2965df7148db60853d7ed0376525915387bcc409))
* **models:** add result model ([ff26736](https://github.com/S1M0N38/llm-fingerprint/commit/ff267367d4e25da7227285f4bebeec515ad57f41))
* **prompts:** add prompts_general_v1.jsonl ([b9ee3cc](https://github.com/S1M0N38/llm-fingerprint/commit/b9ee3cca959498de983540e93a0916b5825605d9))
* **prompts:** add utils for prompts generation ([7d7db2d](https://github.com/S1M0N38/llm-fingerprint/commit/7d7db2d80fe1e16b8abdd6f3e3a1dbe5b790dd77))
* **prompts:** increase the prompt count for general v1 to 8 ([5a4d002](https://github.com/S1M0N38/llm-fingerprint/commit/5a4d0020679f7386dcbe47e78abaa3560138b996))
* **pyproject:** add endpoint to call cli ([60368a8](https://github.com/S1M0N38/llm-fingerprint/commit/60368a88118594eff6e481b7a14b17db3e5db054))
* **query:** add SamplesQuerier ([1efba99](https://github.com/S1M0N38/llm-fingerprint/commit/1efba99c161424143a1cfa3233458b2d44605a66))
* **upload:** implement SampleUploader in upload.py ([1836d4e](https://github.com/S1M0N38/llm-fingerprint/commit/1836d4ecf5972f3eb430b4ca956da3d88374f065))
* **upload:** upsert centroids for model-prompt combinations ([11a2c05](https://github.com/S1M0N38/llm-fingerprint/commit/11a2c059c19518136c0f04a7060714f969283a90))


### Bug Fixes

* **cli:** create parent dirs for --samples-path ([5521466](https://github.com/S1M0N38/llm-fingerprint/commit/55214666dc0132bafaba867945c18319c1fd2677))
* **config:** increase timeout for llama-swap to 2 min ([e95d084](https://github.com/S1M0N38/llm-fingerprint/commit/e95d0846aa70c216d932d94a96bd0f4ddb3b0627))
* **generate:** convert UUID to string in prompt integrity check ([962e9be](https://github.com/S1M0N38/llm-fingerprint/commit/962e9beab6ea19b745a548046c179d3082a0fd57))
* **generate:** enclose generate main coroutine in try/finally ([f5bc117](https://github.com/S1M0N38/llm-fingerprint/commit/f5bc117d5159b750fb7b63b332d3234cff933bc4))
* **generate:** increase the request timeout to 15 min ([652977f](https://github.com/S1M0N38/llm-fingerprint/commit/652977f4ac52e85a7a919cd62b6390a0e5b916ef))
* **justfile:** add missing model for generation ([c49250b](https://github.com/S1M0N38/llm-fingerprint/commit/c49250b9ce7fb0ae8b0faf9a14864fd808b6a2cd))
* **query:** use centroids for query ([c95bb35](https://github.com/S1M0N38/llm-fingerprint/commit/c95bb35b1d31f5d0cabfafdd77c6c5310b9afab8))
* **upload:** check for existing ids and improve prints ([a6b3329](https://github.com/S1M0N38/llm-fingerprint/commit/a6b3329cddde8dc119210afb8ff85934789a2e5d))


### Documentation

* **README:** add readme using llm-thermometer as template ([4d00ab9](https://github.com/S1M0N38/llm-fingerprint/commit/4d00ab992f304a7d67b71b48f1c6f2e930a40cb3))
* **README:** change example collection name ([3804d88](https://github.com/S1M0N38/llm-fingerprint/commit/3804d889025c01084a7a2cee5d79e01d45e6e35b))
* **README:** update cli flags for upload cmd ([bf5fd64](https://github.com/S1M0N38/llm-fingerprint/commit/bf5fd64b257488bb055b174cc6221af2fb8ed287))
* **README:** update env vars in usage section ([c0bd51c](https://github.com/S1M0N38/llm-fingerprint/commit/c0bd51c1709473e886f2d85acac5f2c3976a6c4b))
* **README:** update usage section ([f243bd2](https://github.com/S1M0N38/llm-fingerprint/commit/f243bd21b0c2721e75857002d8d1f69f6c5ea64a))
* remove env vars for local providers ([8abf044](https://github.com/S1M0N38/llm-fingerprint/commit/8abf044f319c2de9b99faf6077caea9337985cc6))
* replace commitizen release with release-please ([7b13c57](https://github.com/S1M0N38/llm-fingerprint/commit/7b13c57bc923b6c1d2da00cb733f767c1abec1ff))
* update .envrc.example ([14dddc8](https://github.com/S1M0N38/llm-fingerprint/commit/14dddc83b5dce7fa01d34c6dd1dae6e71f9eb9d1))
