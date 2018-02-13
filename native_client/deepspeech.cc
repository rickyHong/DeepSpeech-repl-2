#ifdef DS_NATIVE_MODEL
#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "native_client/deepspeech_model_core.h" // generated
#endif

#include <iostream>

#include "deepspeech.h"
#include "deepspeech_utils.h"
#include "alphabet.h"
#include "beam_search.h"

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

#include "c_speech_features.h"

#define BATCH_SIZE 1

#define SAMPLE_RATE 16000

#define AUDIO_WIN_LEN 0.025f
#define AUDIO_WIN_STEP 0.01f
#define AUDIO_WIN_LEN_SAMPLES 400 // AUDIO_WIN_LEN * SAMPLE_RATE
#define AUDIO_WIN_STEP_SAMPLES 160 // AUDIO_WIN_STEP * SAMPLE_RATE

#define MFCC_FEATURES 26
#define MFCC_CONTEXT 9
#define MFCC_WIN_LEN 19 // 2 * MFCC_CONTEXT + 1

#define COEFF 0.97f
#define N_FFT 512
#define N_FILTERS 26
#define LOWFREQ 0
#define CEP_LIFTER 22

using namespace tensorflow;
using tensorflow::ctc::CTCBeamSearchDecoder;
using tensorflow::ctc::CTCDecoder;

namespace DeepSpeech {

struct Private {
  Session* session;
  GraphDef graph_def;
  int ncep;
  int ncontext;
  Alphabet* alphabet;
  KenLMBeamScorer* scorer;
  int beam_width;
  bool run_aot;
};

template<typename T, unsigned int Size, unsigned int Step, bool Print = false>
class Buffer {
public:
  Buffer(std::function<void(T[Size])> callback)
    : callback_(callback)
    , buf_(new T[Size])
    , len_(0)
  {
  }

  ~Buffer()
  {
    delete[] buf_;
  }

  void push(const T* data, unsigned int len)
  {
    if (Print) {printf("push len = %u\n", len);}
    while (len > 0) {
      unsigned int start = len;
      while (len > 0 && len_ < Size) {
        buf_[len_] = *data;
        ++len_;
        ++data;
        --len;
      }
      if (Print) {printf("copied %d to buf, len = %u\n", start - len, len);}

      if (len_ == Size) {
        if (Print) {printf("callback\n");}
        callback_(buf_);
        // shift data by one step
        memmove(buf_, buf_ + Step, Size - Step);
        len_ -= Step;
      }
      if (Print) {printf("buf_ now size %u\n", len_);}
    }
  }

private:
  std::function<void(T[Size])> callback_;
  T* buf_;
  unsigned int len_;
};

struct StreamingState {
  float*** accumulated_logits;
  unsigned int capacity_timesteps;
  unsigned int current_timestep;
  Buffer<short, AUDIO_WIN_LEN_SAMPLES, AUDIO_WIN_STEP_SAMPLES>* audio_buffer;
  Buffer<float, MFCC_FEATURES*MFCC_WIN_LEN, MFCC_FEATURES, true>* mfcc_buffer;
  bool skip_next_mfcc;
};

DEEPSPEECH_EXPORT
StreamingState*
Model::setupStream(unsigned int aPreAllocFrames,
                   unsigned int /*aSampleRate*/)
{
  Status status = mPriv->session->Run({}, {}, {"initialize_state"}, nullptr);

  if (!status.ok()) {
    std::cerr << "Error running session: " << status.ToString() << "\n";
    return nullptr;
  }

  StreamingState* ctx = new StreamingState;
  const size_t num_classes = mPriv->alphabet->GetSize() + 1; // +1 for blank

  float*** logits = (float***)calloc(aPreAllocFrames, sizeof(float**));
  for (int i = 0; i < aPreAllocFrames; ++i) {
    logits[i] = (float**)calloc(BATCH_SIZE, sizeof(float*));
    for (int j = 0; j < BATCH_SIZE; ++j) {
      logits[i][j] = (float*)calloc(num_classes, sizeof(float));
    }
  }

  ctx->accumulated_logits = logits;
  ctx->capacity_timesteps = aPreAllocFrames;
  ctx->current_timestep = 0;

  ctx->audio_buffer = new Buffer<short, AUDIO_WIN_LEN_SAMPLES, AUDIO_WIN_STEP_SAMPLES>([&](short buf[AUDIO_WIN_LEN_SAMPLES]) {
    // Compute MFCC features
    // int n_frames = csf_mfcc(buf, AUDIO_WIN_LEN_SAMPLES, SAMPLE_RATE,
    //                         AUDIO_WIN_LEN, AUDIO_WIN_STEP, MFCC_FEATURES, N_FILTERS, N_FFT,
    //                         LOWFREQ, SAMPLE_RATE/2, COEFF, CEP_LIFTER, 1, NULL,
    //                         &mfcc);
    ctx->mfcc_buffer->push(nullptr, 0);
    // ctx->mfcc_buffer->push(mfcc, n_frames*MFCC_FEATURES);
  });

  ctx->mfcc_buffer = new Buffer<float, MFCC_FEATURES*MFCC_WIN_LEN, MFCC_FEATURES, true>([&](float buf[MFCC_FEATURES*MFCC_WIN_LEN]) {
    printf("mfcc buffer callback\n");
  });

  float initial_zero_context[MFCC_CONTEXT*MFCC_FEATURES] = {};
  // ctx->mfcc_buffer->push(initial_zero_context, MFCC_CONTEXT*MFCC_FEATURES);

  ctx->skip_next_mfcc = false;

  return ctx;
}

DEEPSPEECH_EXPORT
void
Model::feedAudioContent(StreamingState* ctx,
                        const short* aBuffer,
                        unsigned int aBufferSize)
{
  ctx->audio_buffer->push(aBuffer, aBufferSize);
  return;

  // float* mfcc;
  // int n_frames;

  // const size_t num_classes = mPriv->alphabet->GetSize() + 1; // +1 for blank

  // ctx->skip = !ctx->skip;
  // if (!ctx->skip) { // was true
  //   return;
  // }

  // getInputVector(aBuffer, aBufferSize, ctx->sample_rate, &mfcc, &n_frames, nullptr);
  // // assert(n_frames == 1);

  // float*** logits = infer_no_decode(mfcc, 1, 0);

  // if (ctx->current_timestep == ctx->capacity_timesteps) {
  //   unsigned int new_capacity = ctx->capacity_timesteps * 2;
  //   float*** larger = (float***)realloc(ctx->accumulated_logits, new_capacity * sizeof(float**));
  //   for (int i = ctx->current_timestep; i < new_capacity; ++i) {
  //     larger[i] = (float**)calloc(BATCH_SIZE, sizeof(float*));
  //     for (int j = 0; j < BATCH_SIZE; ++j) {
  //       larger[i][j] = (float*)calloc(num_classes, sizeof(float));
  //     }
  //   }
  //   ctx->accumulated_logits = larger;
  //   ctx->capacity_timesteps = new_capacity;
  // }

  // float* dest = &ctx->accumulated_logits[ctx->current_timestep][0][0];
  // for (int i = 0; i < num_classes; ++i) {
  //   dest[i] = logits[0][0][i];
  // }
  // ctx->current_timestep++;

  // free_logits(logits);
}

DEEPSPEECH_EXPORT
char*
Model::finishStream(StreamingState* ctx)
{
  char* str = decode(ctx->current_timestep, ctx->accumulated_logits);
  delete ctx->audio_buffer;
  delete ctx->mfcc_buffer;
  delete ctx;
  return str;
}

DEEPSPEECH_EXPORT
Model::Model(const char* aModelPath, int aNCep, int aNContext,
             const char* aAlphabetConfigPath, int aBeamWidth)
{
  mPriv             = new Private;
  mPriv->session    = nullptr;
  mPriv->scorer     = nullptr;
  mPriv->ncep       = aNCep;
  mPriv->ncontext   = aNContext;
  mPriv->alphabet   = new Alphabet(aAlphabetConfigPath);
  mPriv->beam_width = aBeamWidth;
  mPriv->run_aot    = false;

  if (!aModelPath || strlen(aModelPath) < 1) {
    std::cerr << "No model specified, will rely on built-in model." << std::endl;
    mPriv->run_aot = true;
    return;
  }

  Status status = NewSession(SessionOptions(), &mPriv->session);
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    return;
  }

  status = ReadBinaryProto(Env::Default(), aModelPath, &mPriv->graph_def);
  if (!status.ok()) {
    mPriv->session->Close();
    mPriv->session = nullptr;
    std::cerr << status.ToString() << std::endl;
    return;
  }

  status = mPriv->session->Create(mPriv->graph_def);
  if (!status.ok()) {
    mPriv->session->Close();
    mPriv->session = nullptr;
    std::cerr << status.ToString() << std::endl;
    return;
  }

  for (int i = 0; i < mPriv->graph_def.node_size(); ++i) {
    NodeDef node = mPriv->graph_def.node(i);
    if (node.name() == "logits/shape/1") {
      int final_dim_size = node.attr().at("value").tensor().int_val(0) - 1;
      if (final_dim_size != mPriv->alphabet->GetSize()) {
        std::cerr << "Error: Alphabet size does not match loaded model: alphabet "
                  << "has size " << mPriv->alphabet->GetSize()
                  << ", but model has " << final_dim_size
                  << " classes in its output. Make sure you're passing an alphabet "
                  << "file with the same size as the one used for training."
                  << std::endl;
        mPriv->session->Close();
        mPriv->session = nullptr;
        return;
      }
      break;
    }
  }
}

DEEPSPEECH_EXPORT
Model::~Model()
{
  if (mPriv->session) {
    mPriv->session->Close();
  }

  delete mPriv->alphabet;
  delete mPriv->scorer;

  delete mPriv;
}

DEEPSPEECH_EXPORT
void
Model::enableDecoderWithLM(const char* aAlphabetConfigPath, const char* aLMPath,
                           const char* aTriePath, float aLMWeight,
                           float aWordCountWeight, float aValidWordCountWeight)
{
  mPriv->scorer = new KenLMBeamScorer(aLMPath, aTriePath, aAlphabetConfigPath,
                                      aLMWeight, aWordCountWeight, aValidWordCountWeight);
}

DEEPSPEECH_EXPORT
void
Model::getInputVector(const short* aBuffer, unsigned int aBufferSize,
                      int aSampleRate, float** aMfcc, int* aNFrames,
                      int* aFrameLen)
{
  return audioToInputVector(aBuffer, aBufferSize, aSampleRate, mPriv->ncep,
                            mPriv->ncontext, aMfcc, aNFrames, aFrameLen);
}

char*
Model::decode(int aNFrames, float*** aLogits)
{
  const int batch_size = BATCH_SIZE;
  const int top_paths = 1;
  const int timesteps = aNFrames;
  const size_t num_classes = mPriv->alphabet->GetSize() + 1; // +1 for blank

  // Raw data containers (arrays of floats, ints, etc.).
  int sequence_lengths[batch_size] = {timesteps};

  printf("logits = [\n");
  for (int t = 0; t < timesteps; ++t) {
    printf("[[");
    for (int c = 0; c < num_classes; ++c) {
      printf("%f,", aLogits[t][0][c]);
    }
    printf("]],\n");
  }
  printf("]\n");

  // Convert data containers to the format accepted by the decoder, simply
  // mapping the memory from the container to an Eigen::ArrayXi,::MatrixXf,
  // using Eigen::Map.
  Eigen::Map<const Eigen::ArrayXi> seq_len(&sequence_lengths[0], batch_size);
  std::vector<Eigen::Map<const Eigen::MatrixXf>> inputs;
  inputs.reserve(timesteps);
  for (int t = 0; t < timesteps; ++t) {
    inputs.emplace_back(&aLogits[t][0][0], batch_size, num_classes);
  }

  // Prepare containers for output and scores.
  // CTCDecoder::Output is std::vector<std::vector<int>>
  std::vector<CTCDecoder::Output> decoder_outputs(top_paths);
  for (CTCDecoder::Output& output : decoder_outputs) {
    output.resize(batch_size);
  }
  float score[batch_size][top_paths] = {{0.0}};
  Eigen::Map<Eigen::MatrixXf> scores(&score[0][0], batch_size, top_paths);

  if (mPriv->scorer == nullptr) {
    CTCBeamSearchDecoder<>::DefaultBeamScorer scorer;
    CTCBeamSearchDecoder<> decoder(num_classes,
                                   mPriv->beam_width,
                                   &scorer,
                                   batch_size);
    decoder.Decode(seq_len, inputs, &decoder_outputs, &scores).ok();
  } else {
    CTCBeamSearchDecoder<KenLMBeamState> decoder(num_classes,
                                                 mPriv->beam_width,
                                                 mPriv->scorer,
                                                 batch_size);
    decoder.Decode(seq_len, inputs, &decoder_outputs, &scores).ok();
  }

  // Output is an array of shape (1, n_results, result_length).
  // In this case, n_results is also equal to 1.
  size_t output_length = decoder_outputs[0][0].size() + 1;

  size_t decoded_length = 1; // add 1 for the \0
  for (int i = 0; i < output_length - 1; i++) {
    int64 character = decoder_outputs[0][0][i];
    const std::string& str = mPriv->alphabet->StringFromLabel(character);
    decoded_length += str.size();
  }

  char* output = (char*)malloc(sizeof(char) * decoded_length);
  char* pen = output;
  for (int i = 0; i < output_length - 1; i++) {
    int64 character = decoder_outputs[0][0][i];
    const std::string& str = mPriv->alphabet->StringFromLabel(character);
    strncpy(pen, str.c_str(), str.size());
    pen += str.size();
  }
  *pen = '\0';

  free_logits(aLogits);

  return output;
}

void
Model::free_logits(float*** aLogits)
{
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < BATCH_SIZE; ++j) {
      free(aLogits[i][j]);
    }
    free(aLogits[i]);
  }
  free(aLogits);
}

DEEPSPEECH_EXPORT
float***
Model::infer_no_decode(float* aMfcc, int aNFrames, int aFrameLen)
{
  const int batch_size = BATCH_SIZE;
  const int timesteps = aNFrames;
  const size_t num_classes = mPriv->alphabet->GetSize() + 1; // +1 for blank

  const int frameSize = mPriv->ncep + (2 * mPriv->ncep * mPriv->ncontext);

  assert(timesteps == 1);

  float*** input_data_mat = (float***)calloc(timesteps, sizeof(float**));
  for (int i = 0; i < timesteps; ++i) {
    input_data_mat[i] = (float**)calloc(batch_size, sizeof(float*));
    for (int j = 0; j < batch_size; ++j) {
      input_data_mat[i][j] = (float*)calloc(num_classes, sizeof(float));
    }
  }

  if (mPriv->run_aot) {
#ifdef DS_NATIVE_MODEL
    Eigen::ThreadPool tp(2);  // Size the thread pool as appropriate.
    Eigen::ThreadPoolDevice device(&tp, tp.NumThreads());

    nativeModel nm(nativeModel::AllocMode::RESULTS_AND_TEMPS_ONLY);
    nm.set_thread_pool(&device);

    for (int ot = 0; ot < timesteps; ot += DS_MODEL_TIMESTEPS) {
      nm.set_arg0_data(&(aMfcc[ot * frameSize]));
      nm.Run();

      // The CTCDecoder works with log-probs.
      for (int t = 0; t < DS_MODEL_TIMESTEPS, (ot + t) < timesteps; ++t) {
        for (int b = 0; b < batch_size; ++b) {
          for (int c = 0; c < num_classes; ++c) {
            input_data_mat[ot + t][b][c] = nm.result0(t, b, c);
          }
        }
      }
    }
#else
    std::cerr << "No support for native model built-in." << std::endl;
    return nullptr;
#endif // DS_NATIVE_MODEL
  } else {
    if (aFrameLen != 0 && aFrameLen < frameSize) {
      std::cerr << "mfcc features array is too small (expected " <<
        frameSize << ", got " << aFrameLen << ")\n";
      free_logits(input_data_mat);
      return nullptr;
    }

    Tensor input(DT_FLOAT, TensorShape({1, frameSize}));

    auto input_mapped = input.tensor<float, 2>();
    for (int j = 0, idx = 0; j < frameSize; j++, idx++) {
      input_mapped(0, j) = aMfcc[idx];
    }

    Tensor n_frames(DT_INT32, TensorShape({1}));
    n_frames.scalar<int>()() = aNFrames;

    // The CTC Beam Search decoder takes logits as input
    std::vector<Tensor> outputs;
    Status status = mPriv->session->Run(
      {{ "input_node", input }},
      {"logits"}, {}, &outputs);

    if (!status.ok()) {
      std::cerr << "Error running session: " << status.ToString() << "\n";
      return nullptr;
    }

    auto logits_mapped = outputs[0].tensor<float, 2>();
    // The CTCDecoder works with log-probs.
    for (int t = 0; t < timesteps; ++t) {
      for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < num_classes; ++c) {
          input_data_mat[t][b][c] = logits_mapped(b, c);
        }
      }
    }
  }

  return input_data_mat;
}

DEEPSPEECH_EXPORT
char*
Model::infer(float* aMfcc, int aNFrames, int aFrameLen)
{
  return decode(aNFrames, infer_no_decode(aMfcc, aNFrames, aFrameLen));
}

DEEPSPEECH_EXPORT
char*
Model::stt(const short* aBuffer, unsigned int aBufferSize, int aSampleRate)
{
  float* mfcc;
  char* string;
  int n_frames;

  getInputVector(aBuffer, aBufferSize, aSampleRate, &mfcc, &n_frames, nullptr);
  string = infer(mfcc, n_frames);
  free(mfcc);
  return string;
}

}
