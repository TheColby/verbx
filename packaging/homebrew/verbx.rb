class Verbx < Formula
  include Language::Python::Virtualenv

  desc "Research-grade CLI for extreme reverb processing and analysis"
  homepage "https://github.com/TheColby/verbx"
  url "https://github.com/TheColby/verbx/archive/refs/tags/v0.9.10.tar.gz"
  sha256 "9c7f1dc541530dc3ff3b03614402f35220a434a764d03853e50ff9bed3c40bdf"
  license "MIT"

  depends_on "cmake" => :build
  depends_on "ninja" => :build
  depends_on "llvm@22"
  depends_on "jpeg-turbo"
  depends_on "zlib"
  depends_on "libsndfile"
  depends_on "python@3.14"
  depends_on "numpy"
  depends_on "scipy"
  resource "annotated-doc" do
    url "https://files.pythonhosted.org/packages/57/ba/046ceea27344560984e26a590f90bc7f4a75b06701f653222458922b558c/annotated_doc-0.0.4.tar.gz"
    sha256 "fbcda96e87e9c92ad167c2e53839e57503ecfda18804ea28102353485033faa4"
  end

  resource "audioread" do
    url "https://files.pythonhosted.org/packages/a1/4a/874ecf9b472f998130c2b5e145dcdb9f6131e84786111489103b66772143/audioread-3.1.0.tar.gz"
    sha256 "1c4ab2f2972764c896a8ac61ac53e261c8d29f0c6ccd652f84e18f08a4cab190"
  end

  resource "certifi" do
    url "https://files.pythonhosted.org/packages/af/2d/7bf41579a8986e348fa033a31cdd0e4121114f6bce2457e8876010b092dd/certifi-2026.2.25.tar.gz"
    sha256 "e887ab5cee78ea814d3472169153c2d12cd43b14bd03329a39a9c6e2e80bfba7"
  end

  resource "cffi" do
    url "https://files.pythonhosted.org/packages/eb/56/b1ba7935a17738ae8453301356628e8147c79dbb825bcbc73dc7401f9846/cffi-2.0.0.tar.gz"
    sha256 "44d1b5909021139fe36001ae048dbdde8214afa20200eda0f64c068cac5d5529"
  end

  resource "charset-normalizer" do
    url "https://files.pythonhosted.org/packages/7b/60/e3bec1881450851b087e301bedc3daa9377a4d45f1c26aa90b0b235e38aa/charset_normalizer-3.4.6.tar.gz"
    sha256 "1ae6b62897110aa7c79ea2f5dd38d1abca6db663687c0b1ad9aed6f6bae3d9d6"
  end

  resource "click" do
    url "https://files.pythonhosted.org/packages/3d/fa/656b739db8587d7b5dfa22e22ed02566950fbfbcdc20311993483657a5c0/click-8.3.1.tar.gz"
    sha256 "12ff4785d337a1bb490bb7e9c2b1ee5da3112e94a8622f26a6c77f5d2fc6842a"
  end

  resource "decorator" do
    url "https://files.pythonhosted.org/packages/43/fa/6d96a0978d19e17b68d634497769987b16c8f4cd0a7a05048bec693caa6b/decorator-5.2.1.tar.gz"
    sha256 "65f266143752f734b0a7cc83c46f4618af75b8c5911b00ccb61d0ac9b6da0360"
  end

  resource "idna" do
    url "https://files.pythonhosted.org/packages/6f/6d/0703ccc57f3a7233505399edb88de3cbd678da106337b9fcde432b65ed60/idna-3.11.tar.gz"
    sha256 "795dafcc9c04ed0c1fb032c2aa73654d8e8c5023a7df64a53f39190ada629902"
  end

  resource "lazy-loader" do
    url "https://files.pythonhosted.org/packages/49/ac/21a1f8aa3777f5658576777ea76bfb124b702c520bbe90edf4ae9915eafa/lazy_loader-0.5.tar.gz"
    sha256 "717f9179a0dbed357012ddad50a5ad3d5e4d9a0b8712680d4e687f5e6e6ed9b3"
  end

  resource "librosa" do
    url "https://files.pythonhosted.org/packages/64/36/360b5aafa0238e29758729e9486c6ed92a6f37fa403b7875e06c115cdf4a/librosa-0.11.0.tar.gz"
    sha256 "f5ed951ca189b375bbe2e33b2abd7e040ceeee302b9bbaeeffdfddb8d0ace908"
  end

  resource "markdown-it-py" do
    url "https://files.pythonhosted.org/packages/5b/f5/4ec618ed16cc4f8fb3b701563655a69816155e79e24a17b651541804721d/markdown_it_py-4.0.0.tar.gz"
    sha256 "cb0a2b4aa34f932c007117b194e945bd74e0ec24133ceb5bac59009cda1cb9f3"
  end

  resource "mdurl" do
    url "https://files.pythonhosted.org/packages/d6/54/cfe61301667036ec958cb99bd3efefba235e65cdeb9c84d24a8293ba1d90/mdurl-0.1.2.tar.gz"
    sha256 "bb413d29f5eea38f31dd4754dd7377d4465116fb207585f97bf925588687c1ba"
  end

  resource "msgpack" do
    url "https://files.pythonhosted.org/packages/4d/f2/bfb55a6236ed8725a96b0aa3acbd0ec17588e6a2c3b62a93eb513ed8783f/msgpack-1.1.2.tar.gz"
    sha256 "3b60763c1373dd60f398488069bcdc703cd08a711477b5d480eecc9f9626f47e"
  end

  resource "packaging" do
    url "https://files.pythonhosted.org/packages/65/ee/299d360cdc32edc7d2cf530f3accf79c4fca01e96ffc950d8a52213bd8e4/packaging-26.0.tar.gz"
    sha256 "00243ae351a257117b6a241061796684b084ed1c516a08c48a3f7e147a9d80b4"
  end

  resource "platformdirs" do
    url "https://files.pythonhosted.org/packages/19/56/8d4c30c8a1d07013911a8fdbd8f89440ef9f08d07a1b50ab8ca8be5a20f9/platformdirs-4.9.4.tar.gz"
    sha256 "1ec356301b7dc906d83f371c8f487070e99d3ccf9e501686456394622a01a934"
  end

  resource "pooch" do
    url "https://files.pythonhosted.org/packages/83/43/85ef45e8b36c6a48546af7b266592dc32d7f67837a6514d111bced6d7d75/pooch-1.9.0.tar.gz"
    sha256 "de46729579b9857ffd3e741987a2f6d5e0e03219892c167c6578c0091fb511ed"
  end

  resource "pycparser" do
    url "https://files.pythonhosted.org/packages/1b/7d/92392ff7815c21062bea51aa7b87d45576f649f16458d78b7cf94b9ab2e6/pycparser-3.0.tar.gz"
    sha256 "600f49d217304a5902ac3c37e1281c9fe94e4d0489de643a9504c5cdfdfc6b29"
  end

  resource "pygments" do
    url "https://files.pythonhosted.org/packages/b0/77/a5b8c569bf593b0140bde72ea885a803b82086995367bf2037de0159d924/pygments-2.19.2.tar.gz"
    sha256 "636cb2477cec7f8952536970bc533bc43743542f70392ae026374600add5b887"
  end

  resource "pyloudnorm" do
    url "https://files.pythonhosted.org/packages/23/00/f915eaa75326f4209941179c2b93ac477f2040e4aeff5bb21d16eb8058f9/pyloudnorm-0.2.0.tar.gz"
    sha256 "8bf597658ea4e1975c275adf490f6deb5369ea409f2901f939915efa4b681b16"
  end

  resource "requests" do
    url "https://files.pythonhosted.org/packages/34/64/8860370b167a9721e8956ae116825caff829224fbca0ca6e7bf8ddef8430/requests-2.33.0.tar.gz"
    sha256 "c7ebc5e8b0f21837386ad0e1c8fe8b829fa5f544d8df3b2253bff14ef29d7652"
  end

  resource "rich" do
    url "https://files.pythonhosted.org/packages/b3/c6/f3b320c27991c46f43ee9d856302c70dc2d0fb2dba4842ff739d5f46b393/rich-14.3.3.tar.gz"
    sha256 "b8daa0b9e4eef54dd8cf7c86c03713f53241884e814f4e2f5fb342fe520f639b"
  end

  resource "shellingham" do
    url "https://files.pythonhosted.org/packages/58/15/8b3609fd3830ef7b27b655beb4b4e9c62313a4e8da8c676e142cc210d58e/shellingham-1.5.4.tar.gz"
    sha256 "8dbca0739d487e5bd35ab3ca4b36e11c4078f3a234bfce294b0a0291363404de"
  end

  resource "soundfile" do
    url "https://files.pythonhosted.org/packages/e1/41/9b873a8c055582859b239be17902a85339bec6a30ad162f98c9b0288a2cc/soundfile-0.13.1.tar.gz"
    sha256 "b2c68dab1e30297317080a5b43df57e302584c49e2942defdde0acccc53f0e5b"
  end

  resource "soxr" do
    url "https://files.pythonhosted.org/packages/42/7e/f4b461944662ad75036df65277d6130f9411002bfb79e9df7dff40a31db9/soxr-1.0.0.tar.gz"
    sha256 "e07ee6c1d659bc6957034f4800c60cb8b98de798823e34d2a2bba1caa85a4509"
  end

  resource "typer" do
    url "https://files.pythonhosted.org/packages/f5/24/cb09efec5cc954f7f9b930bf8279447d24618bb6758d4f6adf2574c41780/typer-0.24.1.tar.gz"
    sha256 "e39b4732d65fbdcde189ae76cf7cd48aeae72919dea1fdfc16593be016256b45"
  end

  resource "typing-extensions" do
    url "https://files.pythonhosted.org/packages/72/94/1a15dd82efb362ac84269196e94cf00f187f7ed21c242792a923cdb1c61f/typing_extensions-4.15.0.tar.gz"
    sha256 "0cea48d173cc12fa28ecabc3b837ea3cf6f38c6d1136f85cbaaf598984861466"
  end

  resource "urllib3" do
    url "https://files.pythonhosted.org/packages/c7/24/5f1b3bdffd70275f6661c76461e25f024d5a38a46f04aaca912426a2b1d3/urllib3-2.6.3.tar.gz"
    sha256 "1b62b6884944a57dbe321509ab94fd4d3b307075e0c2eae991ac71ee15ad38ed"
  end

  resource "pillow" do
    url "https://files.pythonhosted.org/packages/ce/a4/9cc30690a394a8305e8be4c9036aa16d0dfddadd09079860b582f562496d/pillow-13.0.0.tar.gz"
    sha256 "cdcd1ba258178f7356d74170c9a81ba4eab6a5d55748bc402a5a2a6fe6be3510"
  end

  resource "llvmlite" do
    url "https://files.pythonhosted.org/packages/11/c5/907cec40688a34eb489cded74d555e1ee4af8cf49d83e03dba2c2d4cfe27/llvmlite-0.50.0.tar.gz"
    sha256 "f2a2cd6ec9ffcc1b7147dea0d7a49efebf17a2b434e0c2844fe175999d571eb4"
  end

  resource "numba" do
    url "https://files.pythonhosted.org/packages/0f/a4/a60b14d955b898ce00d815dfe14435ae3b30c1dd2587aa0a172e92310cf8/numba-0.68.1.tar.gz"
    sha256 "9de461812470173a14eeb7e0bbd27c897dffe835402a33861166842fc5f6ea57"
  end

  resource "joblib" do
    url "https://files.pythonhosted.org/packages/d5/1d/537ab090f302b838943a1b56497dd53059b9a9b46a074936470173a2e207/joblib-1.6.0.tar.gz"
    sha256 "2ccc96785b12046c08fd6d55839c12857831b54a3c1673ffadd2f04bfc4eda03"
  end

  resource "threadpoolctl" do
    url "https://files.pythonhosted.org/packages/56/97/b4748951a683e369513691c8eea33d100298d556308dcb0ab66b132a67f0/threadpoolctl-3.8.0.tar.gz"
    sha256 "01080674ad7ac6182429c918a79de92da789bb448acc02fc0b620b1abd6b6137"
  end

  resource "narwhals" do
    url "https://files.pythonhosted.org/packages/c4/20/e2cce41bfa17909205b39c8a4aa3ec730d87f1c4e77453efd3f5aaf50b81/narwhals-2.29.0.tar.gz"
    sha256 "eb19c877d9a705300bcc9414499a07b32a52db703369414e57e474ca5fc63c42"
  end

  resource "scikit-learn" do
    url "https://files.pythonhosted.org/packages/d2/eb/eaf5e07fcc0da7149b0e084f24e54edd7441b9a89ce7e034032ae97fe3a0/scikit_learn-1.9.1.tar.gz"
    sha256 "629cada3e33e2b9bf376cdc7614a47a4140b8aedc1d836579e359736fbd82977"
  end

  def install
    # Link to the versioned LLVM shared library used by llvmlite.
    ENV["CMAKE_ARGS"] = "-DLLVM_DIR=#{Formula["llvm@22"].opt_lib}/cmake/llvm"
    ENV["LLVMLITE_SHARED"] = "1"
    virtualenv_install_with_resources
  end

  test do
    assert_match "verbx", shell_output("#{bin}/verbx version")
    system Formula["python@3.14"].opt_libexec/"bin/python", "-m", "pip",
           "--python", libexec/"bin/python", "check"
    system libexec/"bin/python", "-c", <<~PYTHON
      import numpy as np
      import librosa
      import numba
      import sklearn
      samples = np.sin(2 * np.pi * 440 * np.arange(4096) / 22050)
      shifted = librosa.effects.pitch_shift(samples, sr=22050, n_steps=2)
      assert shifted.shape == samples.shape
      assert np.isfinite(shifted).all()
      assert np.max(np.abs(shifted)) > 0
    PYTHON
  end
end
