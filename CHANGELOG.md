# Changelog

## [0.5.0](https://github.com/S1M0N38/llm-fingerprint/compare/v0.4.1...v0.5.0) (2025-03-31)


### Features

* **cli:** add --concurrent-requests flag for generate command ([f08d7cd](https://github.com/S1M0N38/llm-fingerprint/commit/f08d7cd46da7e2d613c4ce8278602d31bf4b90db))
* **cli:** add --storage flag for select the storage backend ([8e6b1cc](https://github.com/S1M0N38/llm-fingerprint/commit/8e6b1cc11c5a2f3557a6a37a4efae3719be6ac8f))
* **justfile:** add recipe for generation with 1b models ([9f8e216](https://github.com/S1M0N38/llm-fingerprint/commit/9f8e216dc45c482d210ecc49c074c46fbf06a90a))
* **services:** add base_url and api_key to init of generate ([6544164](https://github.com/S1M0N38/llm-fingerprint/commit/6544164a612149f87c875420b10922ba624b24a3))


### Bug Fixes

* **cli:** add missing --colleciton-name in query command ([3d29ba3](https://github.com/S1M0N38/llm-fingerprint/commit/3d29ba30114da4fac1fd235e8375f8948d2a724c))
* **cli:** move --storage from generate to upload ([493e9b6](https://github.com/S1M0N38/llm-fingerprint/commit/493e9b68467054e85e087269dd88e6fd991fd7ef))
* **justfile:** increase sample size and reduce concurrent reqs ([36f469b](https://github.com/S1M0N38/llm-fingerprint/commit/36f469b9638157d6924b519d98bbf0c9c0206a67))

## [0.4.1](https://github.com/S1M0N38/llm-fingerprint/compare/v0.4.0...v0.4.1) (2025-03-28)


### Bug Fixes

* **storage:** selectively `__init__` EmbeddingMixin ([6fcd901](https://github.com/S1M0N38/llm-fingerprint/commit/6fcd9018caa6e773af5ba1b3ddbddae045ce5bbc))

## [0.4.0](https://github.com/S1M0N38/llm-fingerprint/compare/v0.3.0...v0.4.0) (2025-03-28)


### Features

* **storage:** add optional embeddings parameters to VectorStorage methods ([0277d92](https://github.com/S1M0N38/llm-fingerprint/commit/0277d9211095ed46f473f5456c69369d3a8dfa06))
* **storage:** implement optional embeddings in ChromaStorage ([7f84082](https://github.com/S1M0N38/llm-fingerprint/commit/7f8408229ea3e27f492adf7d4a9e21e0b653a24d))
* **storage:** implement optional embeddings in QdrantStorage ([125d3b5](https://github.com/S1M0N38/llm-fingerprint/commit/125d3b5d378da6c214498a991529912c74d8892b))


### Bug Fixes

* **storage:** accept single embedding in query_sample ([9b5afa2](https://github.com/S1M0N38/llm-fingerprint/commit/9b5afa29379c7b45c1470579b64d1ccb2b883617))
* **storage:** handle optional embeddings in query_samples ([ee60ba8](https://github.com/S1M0N38/llm-fingerprint/commit/ee60ba8a5de3d2b326ee88007f9144f22a006803))


### Documentation

* **CONTRIBUTING:** add release cycle to CONTRIBUTING.md ([1bf6a6a](https://github.com/S1M0N38/llm-fingerprint/commit/1bf6a6a0306f48921a5598ef067016f05ed2afff))

## [0.3.0](https://github.com/S1M0N38/llm-fingerprint/compare/v0.2.0...v0.3.0) (2025-03-25)


### Features

* **qdrant:** add qdrant implementation ([94e7f89](https://github.com/S1M0N38/llm-fingerprint/commit/94e7f8911b7b612940195cf35f2940744a0938d5))


### Bug Fixes

* **chorma:** convert cosine distance to cosine similarity ([48635f0](https://github.com/S1M0N38/llm-fingerprint/commit/48635f0031a4661b3000ec25f56bec177d94a235))
* **chroma:** set cosine distance as distance function ([b167eae](https://github.com/S1M0N38/llm-fingerprint/commit/b167eae8b7cf6533af4ae4700b9c774bf7083531))
* **chroma:** use cosine for distance function ([131ac73](https://github.com/S1M0N38/llm-fingerprint/commit/131ac73ffbe10b732c70a12335e46ddf795f5048))
* use uuid.uuid4 for Sample id ([d9785b8](https://github.com/S1M0N38/llm-fingerprint/commit/d9785b8c47a7e79487abf15ee9dd7beb66edd724))


### Documentation

* **CONTRIBUTING:** update diagrams ([56fec4c](https://github.com/S1M0N38/llm-fingerprint/commit/56fec4c8b9c3edf6a4d35bce2560a2bb9328ba4f))
* **envrc:** add qdrant env var to .envrc.example ([df6f641](https://github.com/S1M0N38/llm-fingerprint/commit/df6f641a6f0b8f7f544d362662cf4f172b9c3648))

## [0.2.0](https://github.com/S1M0N38/llm-fingerprint/compare/v0.1.1...v0.2.0) (2025-03-24)


### Features

* **io:** add method to load results to FileIO ([34c9ecb](https://github.com/S1M0N38/llm-fingerprint/commit/34c9ecba4456627127be4a037ea3f6787703eab5))
* move from chromadb ef to openai embedding url ([0ad6a70](https://github.com/S1M0N38/llm-fingerprint/commit/0ad6a70dc385d2f1a4b2e466520f25547015e182))


### Bug Fixes

* add results_num to QuerierService ([4b82c5b](https://github.com/S1M0N38/llm-fingerprint/commit/4b82c5be19d639a2ea0870d711971179f5562809))
* **chroma:** results_num args ([08701bd](https://github.com/S1M0N38/llm-fingerprint/commit/08701bd68a706aae0e6fd9a9c9d1a321dbd54207))
* **cli:** update cli cmd to use embedding model ([78eedab](https://github.com/S1M0N38/llm-fingerprint/commit/78eedabe2cbbed504fcb68acc874b564993be864))
* **mixin:** remove `super().__init__` in mixin classes ([bae3df5](https://github.com/S1M0N38/llm-fingerprint/commit/bae3df5f7af94ffd8af8e61730a61d1f667199a1))
* **query:** move chroma init to initialize and add print statements ([2712255](https://github.com/S1M0N38/llm-fingerprint/commit/27122558665218ce8fb492e6bf1a9b0a6b56cfe7))
* remove tight coupling of env var and class init ([a8608ab](https://github.com/S1M0N38/llm-fingerprint/commit/a8608abe87f5672ff7acafbf6e18718035b6efe1))
* typo in function names and args ([861bf71](https://github.com/S1M0N38/llm-fingerprint/commit/861bf715074fe6096de4b7eb1da85cbc2be64f61))


### Documentation

* **CONTRIBUTING:** add diagrams ([8131e5f](https://github.com/S1M0N38/llm-fingerprint/commit/8131e5f29859d790a262df87c3a190738c309319))
* **CONTRIBUTING:** update diagrams ([6f6c836](https://github.com/S1M0N38/llm-fingerprint/commit/6f6c836ff3c3294a4fc568c0e3166a19afc68016))
* update env var to make use of embedding endpoint ([f795a72](https://github.com/S1M0N38/llm-fingerprint/commit/f795a721479f2c116b7ea5b3e881de66f46dd6c7))

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
