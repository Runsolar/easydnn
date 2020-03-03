//Under construction, it doesn't work yet
/*
This code is devoted to my dear father Jesus Christ.
Code by St. Spirit and Danijar Wolf, Feb 20, 2020.
*/

#include <iostream>
#include<ctime>

#define DEFAULT_LEARNINGRATE 0.4
#define DEFAULT_BIASLEARNINGRATE 0.03
//#define DEFAULT_MOMENTUM 0.1

template <typename T> struct DataSet;
enum class Activation { SIGMOID, SOFTMAX, RELU, PLAIN };
template<typename T> class Neuron;
template<typename T> class Matrix;
template<typename T> class Layer;
template<class U, typename T> class NeuralNetwork;

template<typename T>
const T Sigmoid(const T& x)
{
    return static_cast<T>(1) / (1 + exp(-x));
}

template<typename T>
const T Relu(const T& x)
{
    return (x < 0)? static_cast<T>(0) : x;
}


template<typename T>
class Neuron {
    friend Layer<T>;
    //friend const Neuron<T> operator-(const Neuron<T>& x, const Neuron<T>& y);
    friend NeuralNetwork<Layer<T>, T>;

public:

    Neuron() = delete;

    explicit Neuron(const unsigned len) : len(len), array(nullptr) {
        try {
            array = new T[len]();
            //std::cout << "A new Neuron hase been created... " << this << std::endl;
        }
        catch (std::exception & ex) {
            std::cout << "En exception is happened... " << ex.what() << std::endl;
            return;
        }
    }

    Neuron(const Neuron<T>& vec): Neuron(vec.len) {
        if (this == &vec) return;
        for (unsigned i = 0; i < len; ++i) {
            array[i] = vec[i];
        }
    }

    ~Neuron() {
        delete[] array;
        //std::cout << "A Neuron hase been deleted... " << this << std::endl;
    }

    T& operator[](const unsigned index) {
        return array[index];
    }

    const T& operator[](const unsigned index) const {
        return array[index];
    }

    const T dot(const Neuron<T>& neuron) const;
    const Neuron<T> operator*(const Matrix<T>& matrix) const;
    const Neuron<T> operator*(const Neuron<T>& neuron) const;

    const Neuron<T>& operator-=(const Neuron<T>& neuron) const;
    const Neuron<T> operator-(const Neuron<T>& neuron) const;

    Neuron<T>& operator=(const Neuron<T>& neuron);

private:
    unsigned len;
    T* array;

    const Neuron<T> reluDerivativeFunc() const;
};


template<typename T>
const T Neuron<T>::dot(const Neuron<T>& neuron) const {
    T sum = 0;
    if (len == neuron.len) {
        for (unsigned i = 0; i < len; ++i) {
            sum += array[i] * neuron[i];
        }
    }
    return sum;
}

template <typename T>
const Neuron<T> Neuron<T>::reluDerivativeFunc() const {
    Neuron<T> vec(this->len);
    for (unsigned i = 0; i < vec.len; ++i) {
        vec[i] = (this->array[i] < static_cast<T>(0)) ? static_cast<T>(0) : static_cast<T>(1);
    }
    return vec;
}

template<typename T>
const Neuron<T> Neuron<T>::operator*(const Matrix<T>& matrix) const {
    Neuron<T> vec(matrix.cols);
    for (unsigned i = 0; i < matrix.cols; ++i) {
        vec[i] = dot(matrix[i]);
    }
    return vec;
}

template<typename T>
const Neuron<T> Neuron<T>::operator*(const Neuron<T>& neuron) const {
    Neuron<T> vec(neuron.len);
    for (unsigned i = 0; i < len; ++i) {
        vec[i] = array[i] * neuron[i];
    }
    return vec;
}

template<typename T>
const Neuron<T> Neuron<T>::operator-(const Neuron<T>& neuron) const {
    Neuron<T> vec(*this);
    vec -= neuron;
    return vec;
}

template<typename T>
const Neuron<T>& Neuron<T>::operator-=(const Neuron<T>& neuron) const {
    //this += Neuron * (-1);
    if (len == neuron.len) {
        for (unsigned i = 0; i < len; ++i) {
            array[i] -= neuron[i];
        }
    }
    return *this;
}

template<typename T>
Neuron<T>& Neuron<T>::operator=(const Neuron<T>& neuron) {
    if (this == &neuron) return *this;
    if (len == neuron.len) {
        for (unsigned i = 0; i < len; ++i) {
            array[i] = neuron[i];
        }
    }
    else {
        len = neuron.len;
        if(array != nullptr) delete[] array;
        array = new T[len]();
        for (unsigned i = 0; i < len; ++i) {
            array[i] = neuron[i];
        }
    }
    return *this;
}


template<typename T>
class Matrix {
    friend Neuron<T>;
    friend Layer<T>;
    friend NeuralNetwork<Layer<T>, T>;

public:
    Matrix() = delete;

    explicit Matrix(const unsigned rows, const unsigned cols) : rows(rows), cols(cols) {
        matrix = new Neuron<T> * [cols];
        for (unsigned i = 0; i < cols; ++i) {
            matrix[i] = new Neuron<T>(rows);
        }
        //std::cout << "A new matrix has been created... " << this << std::endl;
    }

    Matrix(const Matrix<T>& matrixObj): Matrix(matrixObj.rows, matrixObj.cols) {
        for (unsigned i = 0, j; i < cols; ++i) {
            Neuron<T>& col = *matrix[i];
            for (j = 0; j < rows; j++) {
                col[j] = matrixObj[i][j];
            }
        }
    }
/*
    T& at(const unsigned i, const unsigned j) {
        Neuron<T>& Col = *matrix[i];
        return Col[j];
    }
*/
    Neuron<T>& operator[](const unsigned index) const {
        return *matrix[index];
    }

    Matrix<T> operator*(const T& scalar) {
        Matrix<T> res(rows, cols);

        for (unsigned j = 0, i; j < rows; ++j) {
            for (i = 0; i < cols; ++i) {
                res[i][j] = (*matrix[i])[j] * scalar;
            }
        }

        return res;
    }

    Neuron<T> operator*(const Neuron<T>& vec) {
        Neuron<T> res(rows);

        for (unsigned j = 0, i; j < rows; ++j) {
            res[j] = 0;
            for (i = 0; i < cols; ++i) {
                res[j] += (*matrix[i])[j] * vec[i];
            }
        }

        return res;
    }

    Matrix<T> operator*(const Matrix<T>& m) {
        Matrix<T> res(rows, m.cols);

        for (unsigned j = 0, i, k; j < rows; ++j) {
            for (i = 0; i < m.cols; ++i) {
                res[i][j] = 0;
                for (k = 0; k < cols; ++k) {
                    res[i][j] = res[i][j] + (*matrix[k])[j] * m[i][k];
                }
            }
        }

        return res;
    }

    Matrix<T>& operator-=(const Matrix<T>& m) {
        for (unsigned j = 0, i; j < rows; ++j) {
            for (i = 0; i < cols; ++i) {
                (*matrix[i])[j] -= m[i][j];
            }
        }
        return *this;
    }

    Matrix<T> operator-(const Matrix<T>& m) {
        Matrix<T> res(rows, cols);
        
        res = *this;
        res -= m;

        return res;
    }

    Matrix<T>& operator=(const Matrix<T>& matrixObj) {
        if (cols == matrixObj.cols && rows == matrixObj.rows) {
            for (unsigned i = 0, j; i < cols; ++i) {
                Neuron<T>& col = *matrix[i];
                for (j = 0; j < rows; j++) {
                    col[j] = matrixObj[i][j];
                }
            }
        }
        return *this;
    }

    ~Matrix() {
        for (unsigned i = 0; i < cols; ++i) {
            delete matrix[i];
        }
        delete[] matrix;
        //std::cout << "A matrix has been deleted... " << this << std::endl;
    }

private:
    const unsigned cols;
    const unsigned rows;
    Neuron<T>** matrix;
};


template<typename T>
class Layer {
    friend NeuralNetwork<Layer<T>, T>;

public:
    T bias;    // bias of each layer
    //using element_type = typename std::remove_reference< decltype(std::declval<T>) >::type;
    Neuron<T> outputs;
    Activation transferFunction;

    Layer(const unsigned numsOfWeights,
        const unsigned numsOfPerceptrons, Activation transferFunction) :
        rows(numsOfWeights),
        cols(numsOfPerceptrons),
        transferFunction(transferFunction),
        weights(rows, cols),
        outputs(cols),
        bias(static_cast<T>(1)) {
        for (unsigned i = 0, j; i < cols; ++i) {
            for (j = 0; j < rows; ++j) {
                weights[i][j] = static_cast<unsigned>(rand() % 2) ? static_cast<T>(rand()) / RAND_MAX : static_cast<T>(rand()) / -RAND_MAX;
            }
        }
    }

    ~Layer() {}

    void PrunsignedLayer() const {
        for (unsigned j = 0, i; j < rows; ++j) {
            for (i = 0; i < cols; ++i) {
                std::cout << static_cast<T>(weights[i][j]) << "   ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    void PrunsignedOutputs() const {
        for (unsigned i = 0; i < outputs.len; ++i) {
            std::cout << outputs[i] << std::endl;
        }
    }

    Neuron<T> BackPropagation(const Neuron<T>& input, const Neuron<T>& inputs, const T& learning_rate);
    void FeedForward(const Neuron<T>&);

private:
    const unsigned cols;
    const unsigned rows;
    Matrix<T> weights;

    //const 
    void activation_mapper();
};

template<typename T>
Neuron<T> Layer<T>::BackPropagation(const Neuron<T>& errors, const Neuron<T>& input, const T& learning_rate) {
    T bias_Delta = static_cast<T>(1.0);
    //Neuron<T> derOfSigmoid(outputs.len);
    Neuron<T> gamma(outputs.len);

    //derOfSigmoid = outputs - outputs*outputs;
    switch (transferFunction) {
    case Activation::SIGMOID:
        gamma = errors * (outputs - outputs * outputs);
        break;
    case Activation::SOFTMAX:
        break;
    case Activation::RELU:
        gamma = errors *  outputs.reluDerivativeFunc();
        break;
    case Activation::PLAIN:
        gamma = errors * outputs;
        break;
    }

    Matrix<T> in(input.len, 1);
    Matrix<T> wdl(1, gamma.len);
    //Matrix<T> gradients(in.rows, wdl.cols);
    
    in[0] = input;
    for(unsigned i=0; i< gamma.len; ++i) wdl[i][0] = gamma[i];
    //gradients = in * wdl;

    weights -= in * wdl * learning_rate;

    for (unsigned i = 0; i < gamma.len; ++i) {
        bias_Delta *= gamma[i];
    }
    bias -= bias_Delta * DEFAULT_BIASLEARNINGRATE;
    //bias -= bias_Delta * learning_rate;

    return gamma;
}

template<typename T>
void Layer<T>::FeedForward(const Neuron<T>& input) {
    outputs = input * weights;
    activation_mapper();
}

template<typename T>
void Layer<T>::activation_mapper() {
    for (unsigned i = 0; i < outputs.len; ++i) {

        switch (transferFunction) {
        case Activation::SIGMOID:
            outputs[i] = Sigmoid(outputs[i] + bias);
            break;
        case Activation::SOFTMAX:
            break;
        case Activation::RELU:
            outputs[i] = Relu(outputs[i] + bias);
            break;
        case Activation::PLAIN:
            outputs[i] = outputs[i] + bias;
            break;
        }

    }
}


template<class U, typename T>
class NeuralNetwork {
public:
    T learning_rate;
    mutable T m_error;
    Neuron<T> result;

    using element_type = typename std::remove_reference< decltype(std::declval<U>()) >::type;

    explicit NeuralNetwork() : head(nullptr), tail(nullptr), learning_rate(DEFAULT_LEARNINGRATE) {
        std::cout << "The NeuralNetwork has been created" << this << std::endl;
    }

    explicit NeuralNetwork(const T learning_rate) : head(nullptr), tail(nullptr), 
        learning_rate(learning_rate), 
        m_error(static_cast<T>(0)), 
        result(0), 
        pDataSet(nullptr){
        std::cout << "The NeuralNetwork has been created" << this << std::endl;
    }

    ~NeuralNetwork();

    void pushLayer(U& layerObj);
    void mountDataSet(const DataSet<T>& dataset);
    void Train(const unsigned epochs) const;

private:
    void BackPropagation(const Neuron<T>& input, const Neuron<T>& label) const;
    void FeedForward(const Neuron<T>& input) const;
    void getResult() const;
    const void predict(const Neuron<T>& input) const;

    template<class U>
    class Domain {
    public:
        U& layer;
        Domain<U>* pNextDomain;
        Domain<U>* pPreviousDomain;
        Domain(U& layer = U(), Domain<U>* pPreviousDomain = nullptr, Domain<U>* pNextDomain = nullptr) :
            layer(layer),
            pNextDomain(pNextDomain),
            pPreviousDomain(pPreviousDomain) {}
    };
    Domain<U>* head;
    Domain<U>* tail;
    const DataSet<T>* pDataSet;
};

template<class U, typename T>
NeuralNetwork<U, T>::~NeuralNetwork() {
    if (head != nullptr) {
        while (head != nullptr) {
            Domain<U>* current = head;
            head = current->pNextDomain;
            delete current;
            std::cout << "A Domain has been deleted... " << this << std::endl;
        }
        std::cout << "The NeuralNetwork has been deleted... " << this << std::endl;
    }
}

template<class U, typename T>
void NeuralNetwork<U, T>::pushLayer(U& layerObj) {
    if (head == nullptr) {
        head = new Domain<U>(layerObj);
        std::cout << "A first Domain has been created..." << this << std::endl;
        return;
    }

    Domain<U>* current = head;
    while (current->pNextDomain != nullptr) {
        current = current->pNextDomain;
    }
    current->pNextDomain = new Domain<U>(layerObj, current);
    tail = current->pNextDomain;

    std::cout << "A Domain has been created..." << this << std::endl;
    return;
}

template<class U, typename T>
void NeuralNetwork<U, T>::BackPropagation(const Neuron<T>& input, const Neuron<T>& label) const {
    Domain<U>* current = tail;
    Neuron<T> errors(label);
    const Neuron<T>* pInput;
    U* pLayer;
    U* pPreviousLayer;

    pLayer = &current->layer;
    while (current != nullptr) {

        pLayer = &current->layer;
        if (current->pNextDomain == nullptr) {
            errors = pLayer->outputs - label;
        }
        if (current->pPreviousDomain != nullptr) {
            pPreviousLayer = &current->pPreviousDomain->layer; 
            pInput = &pPreviousLayer->outputs;
        }
        else {
            pInput = &input;
        }

        errors = pLayer->BackPropagation(errors, *pInput, learning_rate);
        errors = pLayer->weights * errors;
        current = current->pPreviousDomain;

        //pLayer->PrunsignedLayer();
        //std::cout << std::endl;
    }
};

template<class U, typename T>
void NeuralNetwork<U, T>::FeedForward(const Neuron<T>& input) const {
    Domain<U>* current = head;
    const Neuron<T>* pInput;
    pInput = &input;
    
    U* pLayer;
    while (current != nullptr) {

        pLayer = &current->layer;
        pLayer->FeedForward(*pInput);
        pInput = &pLayer->outputs;
        current = current->pNextDomain;
        
        //pLayer->PrunsignedOutputs();
        //std::cout << std::endl;
        //pLayer->PrunsignedLayer();
        //std::cout << std::endl;
    }
}

template<class U, typename T>
void NeuralNetwork<U, T>::mountDataSet(const DataSet<T> &dataset) {
    pDataSet =  &dataset;
};

template<class U, typename T>
void NeuralNetwork<U, T>::Train(const unsigned epochs) const {
    Domain<U>* current = tail;
    U* pLayer = &current->layer;

    //result = pLayer->outputs;

    T r;
    T delta_err;

    //std::cout << &pDataSet->inputs.cols << std::endl;
    //std::cout << &pDataSet->labels.rows << std::endl;

    Neuron<T> input(pDataSet->inputs.cols);
    Neuron<T> label(pDataSet->labels.rows);

    for (unsigned epoch = 0; epoch < epochs; ++epoch) {
        for (unsigned j = 0; j < pDataSet->inputs.rows; ++j)
        {
            for (unsigned i = 0; i < pDataSet->inputs.cols; ++i) {
                input[i] = pDataSet->inputs[i][j];
            }

            for (unsigned i = 0; i < pDataSet->labels.rows; ++i) {
                label[i] = pDataSet->labels[j][i];
            }

            FeedForward(input);
            BackPropagation(input, label);

            m_error = static_cast<T>(0);

            for (unsigned i = 0; i < (pLayer->outputs).len; ++i) {
                delta_err = pLayer->outputs[i] - label[i];
                m_error += delta_err * delta_err;
            }

            //m_error = m_error / ((pLayer->outputs).len);
            r = ((pLayer->outputs).len > 1) ? (pLayer->outputs).len - 1 : 1;
            m_error = m_error / r;
            m_error = static_cast<T>(sqrt(m_error));

            std::cout << "epoch is: " << epoch << std::endl;
            printf("error = %f\r\n", m_error);

        }
    }
};

template <typename U, typename T>
void NeuralNetwork<U, T>::getResult() const {
    U* pLayer = tail;
    result = pLayer->outputs;
}

template <typename U, typename T>
const void NeuralNetwork<U, T>::predict(const Neuron<T>& input) const {
    FeedForward(input);
}


template <typename T>
struct DataSet {
    DataSet(const Matrix<T>& inputs, const Matrix<T>& labels): inputs(inputs), labels(labels) {};
    const Matrix<T>& inputs;
    const Matrix<T>& labels;
};


unsigned main()
{

    const double _inputs[8][3] = {
      {0, 0, 0}, //0
      {0, 0, 1}, //1
      {0, 1, 0}, //1
      {0, 1, 1}, //0
      {1, 0, 0}, //1
      {1, 0, 1}, //0
      {1, 1, 0}, //0
      {1, 1, 1}  //1
    };
    

   const double _expectedLabels[8][1] = { {0}, {1}, {1}, {0}, {1}, {0}, {0}, {1} };


    //srand(time(NULL));

    const Matrix<double> inputs(8, 3);
    inputs[0][0] = 0; inputs[1][0] = 0; inputs[2][0] = 0;
    inputs[0][1] = 0; inputs[1][1] = 0; inputs[2][1] = 1;
    inputs[0][2] = 0; inputs[1][2] = 1; inputs[2][2] = 0;
    inputs[0][3] = 0; inputs[1][3] = 1; inputs[2][3] = 1;
    inputs[0][4] = 1; inputs[1][4] = 0; inputs[2][4] = 0;
    inputs[0][5] = 1; inputs[1][5] = 0; inputs[2][5] = 1;
    inputs[0][6] = 1; inputs[1][6] = 1; inputs[2][6] = 0;
    inputs[0][7] = 1; inputs[1][7] = 1; inputs[2][7] = 1;


    const Matrix<double> expectedLabels(1, 8);
    expectedLabels[0][0] = 0; expectedLabels[1][0] = 1; expectedLabels[2][0] = 1;  expectedLabels[3][0] = 0;
    expectedLabels[4][0] = 1; expectedLabels[5][0] = 0; expectedLabels[6][0] = 0;  expectedLabels[7][0] = 1;

    Layer<double> layer1(3, 3, Activation::SIGMOID);
    Layer<double> layer2(3, 9, Activation::SIGMOID);
    Layer<double> layer3(9, 9, Activation::SIGMOID);
    Layer<double> layer4(9, 1, Activation::SIGMOID);

    NeuralNetwork<Layer<double>, double> NeuralNetwork(0.3);
    NeuralNetwork.pushLayer(layer1);
    NeuralNetwork.pushLayer(layer2);
    NeuralNetwork.pushLayer(layer3);
    NeuralNetwork.pushLayer(layer4);

/*
    Neuron<double> input(3);
    input[0] = 0;
    input[1] = 0;
    input[2] = 0;

    NeuralNetwork.FeedForward(input);
*/
    DataSet<double> dataset(inputs, expectedLabels);
    NeuralNetwork.mountDataSet(dataset);
    NeuralNetwork.Train(100);


    //NeuralNetwork.pushLayer(layer0);
    //NeuralNetwork.pushLayer(Layer<double>(120, 64));
    //NeuralNetwork.pushLayer(Layer<double>(64, 32));
    //NeuralNetwork.pushLayer(Layer<double>(32, 16));
    //NeuralNetwork.pushLayer(Layer<double>(16, 10));


    std::cout << "Hello World!\n";
}
