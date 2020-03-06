//Under construction, it doesn't work yet
/*
This code is devoted to my dear father Jesus Christ.
Code by St. Spirit and Danijar Wolf, Feb 20, 2020.
*/

#include <iostream>
#include<ctime>
#include <random>

#define DEFAULT_LEARNINGRATE 0.15
#define DEFAULT_BIASLEARNINGRATE 0.05
#define DEFAULT_MOMENTUM 0.2

template <typename T> struct DataSet;
enum class Activation { SIGMOID, SOFTMAX, RELU, PLAIN };
template<typename T> class Vector;
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
    return (x < 0) ? static_cast<T>(0) : x;
}


template<typename T>
class Vector {
    friend Layer<T>;
    //friend const Vector<T> operator-(const Vector<T>& x, const Vector<T>& y);
    friend NeuralNetwork<Layer<T>, T>;

public:
    unsigned len;

    //Vector() = delete;
    Vector() : len(0), array(nullptr) {}

    explicit Vector(const unsigned &len) : len(len), array(nullptr) {
        try {
            array = new T[len]();
            //std::cout << "A new Vector hase been created... " << this << std::endl;
        }
        catch (std::exception & ex) {
            std::cout << "En exception is happened... " << ex.what() << std::endl;
            return;
        }
    }

    Vector(const Vector<T>& vec) : Vector(vec.len) {
        if (this == &vec) return;
        for (unsigned i = 0; i < len; ++i) {
            array[i] = vec[i];
        }
    }

    ~Vector() {
        delete[] array;
        //std::cout << "A Vector hase been deleted... " << this << std::endl;
    }

    const unsigned getLen() const {
        return len;
    }

    T& operator[](const unsigned index) {
        return array[index];
    }

    const T& operator[](const unsigned index) const {
        return array[index];
    }

    const T dot(const Vector<T>& Vector) const;
    const Vector<T> operator*(const Matrix<T>& matrix) const;
    const Vector<T> operator*(const Vector<T>& Vector) const;
    const Vector<T> operator*(const T& scalar) const;
    const Vector<T>& operator-=(const Vector<T>& Vector) const;
    const Vector<T> operator-(const Vector<T>& Vector) const;

    Vector<T>& operator=(const Vector<T>& Vector);
    Vector<T>& operator()(const Vector<T>& Vector);

protected:
    T* array;
    const Vector<T> reluDerivativeFunc() const;
};


template<typename T>
const T Vector<T>::dot(const Vector<T>& Vector) const {
    T sum = 0;
    if (len == Vector.len) {
        for (unsigned i = 0; i < len; ++i) {
            sum += array[i] * Vector[i];
        }
    }
    return sum;
}

template <typename T>
const Vector<T> Vector<T>::reluDerivativeFunc() const {
    Vector<T> vec(len);
    for (unsigned i = 0; i < len; ++i) {
        vec[i] = (this->array[i] < static_cast<T>(0)) ? static_cast<T>(0) : static_cast<T>(1);
    }
    return vec;
}

template<typename T>
const Vector<T> Vector<T>::operator*(const Matrix<T>& matrix) const {
    Vector<T> vec(matrix.cols);
    for (unsigned i = 0; i < matrix.cols; ++i) {
        vec[i] = dot(matrix[i]);
    }
    return vec;
}

template<typename T>
const Vector<T> Vector<T>::operator*(const Vector<T>& vector) const {
    Vector<T> vec(len);
    for (unsigned i = 0; i < len; ++i) {
        vec[i] = array[i] * vector[i];
    }
    return vec;
}

template<typename T>
const Vector<T> Vector<T>::operator*(const T& scalar) const {
    Vector<T> vec(len);
    for (unsigned i = 0; i < len; ++i) {
        vec[i] = array[i] * scalar;
    }
    return vec;
}

template<typename T>
const Vector<T> Vector<T>::operator-(const Vector<T>& vector) const {
    Vector<T> vec(*this);
    vec -= vector;
    return vec;
}

template<typename T>
const Vector<T>& Vector<T>::operator-=(const Vector<T>& vector) const {
    //this += Vector * (-1);
    if (len == vector.len) {
        for (unsigned i = 0; i < len; ++i) {
            array[i] -= vector[i];
        }
    }
    return *this;
}

template<typename T>
Vector<T>& Vector<T>::operator=(const Vector<T>& vector) {
    if (this == &vector) return *this;
    if (len == vector.len) {
        for (unsigned i = 0; i < len; ++i) {
            array[i] = vector[i];
        }
    }
    else {
        len = vector.len;
        if (array != nullptr) delete[] array;
        array = new T[len]();
        for (unsigned i = 0; i < len; ++i) {
            array[i] = vector[i];
        }
    }
    return *this;
}

template<typename T>
Vector<T>& Vector<T>::operator()(const Vector<T>& vector) {

    if (array != nullptr) {
        delete[] array;
    }
    len = vector.len;

    array = new T[len]();

    for (unsigned i = 0; i < len; ++i)
    {
        array[i] = vector[i];
    }

    return *this;
}

template<typename T>
class Matrix {
    friend Vector<T>;
    friend Layer<T>;
    friend NeuralNetwork<Layer<T>, T>;

public:
    //Matrix() = delete;
    Matrix(): rows(0), cols(0), matrix(nullptr) {}

    explicit Matrix(const unsigned rows, const unsigned cols) : rows(rows), cols(cols) {
        matrix = new Vector<T> * [cols];
        for (unsigned i = 0; i < cols; ++i) {
            matrix[i] = new Vector<T>(rows);
        }
        //std::cout << "A new matrix has been created... " << this << std::endl;
    }

    Matrix(const Matrix<T>& matrixObj) : Matrix(matrixObj.rows, matrixObj.cols) {
        for (unsigned i = 0, j; i < cols; ++i) {
            Vector<T>& col = *matrix[i];
            for (j = 0; j < rows; j++) {
                col[j] = matrixObj[i][j];
            }
        }
    }
    /*
        T& at(const unsigned i, const unsigned j) {
            Vector<T>& Col = *matrix[i];
            return Col[j];
        }
    */
    Vector<T>& operator[](const unsigned index) const {
        return *matrix[index];
    }

    const Matrix<T> operator*(const T& scalar) const;
    const Vector<T> operator*(const Vector<T>& vec) const;
    const Matrix<T> operator*(const Matrix<T>& m) const;
    const Matrix<T>& operator-=(const Matrix<T>& m) const;
    const Matrix<T> operator-(const Matrix<T>& m) const;

    const Matrix<T>& operator=(const Matrix<T>& matrixObj) const {
        if (cols == matrixObj.cols && rows == matrixObj.rows) {
            for (unsigned i = 0, j; i < cols; ++i) {
                Vector<T>& col = *matrix[i];
                for (j = 0; j < rows; j++) {
                    col[j] = matrixObj[i][j];
                }
            }
        }
        return *this;
    }

    Matrix<T>& operator()(const Matrix<T>& matrix);

    ~Matrix() {
        for (unsigned i = 0; i < cols; ++i) {
            delete matrix[i];
        }
        delete[] matrix;
        //std::cout << "A matrix has been deleted... " << this << std::endl;
    }

private:
    unsigned cols;
    unsigned rows;
    Vector<T>** matrix;
};


template<typename T>
const Matrix<T> Matrix<T>::operator*(const T& scalar) const {
    Matrix<T> res(rows, cols);

    for (unsigned j = 0, i; j < rows; ++j) {
        for (i = 0; i < cols; ++i) {
            res[i][j] = (*matrix[i])[j] * scalar;
        }
    }
    return res;
}

template<typename T>
const Vector<T> Matrix<T>::operator*(const Vector<T>& vec) const {
    Vector<T> res(rows);

    for (unsigned j = 0, i; j < rows; ++j) {
        res[j] = 0;
        for (i = 0; i < cols; ++i) {
            res[j] += (*matrix[i])[j] * vec[i];
        }
    }
    return res;
}

template<typename T>
const Matrix<T> Matrix<T>::operator*(const Matrix<T>& m) const {
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

template<typename T>
const Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& m) const {
    for (unsigned j = 0, i; j < rows; ++j) {
        for (i = 0; i < cols; ++i) {
            (*matrix[i])[j] -= m[i][j];
        }
    }
    return *this;
}

template<typename T>
const Matrix<T> Matrix<T>::operator-(const Matrix<T>& m) const {
    Matrix<T> res(rows, cols);

    res = *this;
    res -= m;

    return res;
}

template<typename T>
Matrix<T>& Matrix<T>::operator()(const Matrix<T>& matrixObj) {
    cols = matrixObj.cols;
    rows = matrixObj.rows;

    if (matrix != nullptr) {
        delete[] matrix;
    }

    matrix = new Vector<T> * [cols];
    for (unsigned i = 0; i < cols; ++i) {
        matrix[i] = new Vector<T>(rows);
    }

    for (unsigned i = 0, j; i < cols; ++i) {
        Vector<T>& col = *matrix[i];
        for (j = 0; j < rows; j++) {
            col[j] = matrixObj[i][j];
        }
    }

    return *this;
}




template<typename T>
class Neuron : public Vector<T> {
public:
    unsigned numOfInputs = 0;

    Neuron(): Vector<T>::Vector(), numOfInputs(0), state(0), weights(nullptr) {}

    explicit Neuron(const unsigned &numOfImputs) : Vector<T>::Vector(numOfInputs) {
        this->numOfInputs = numOfImputs;
        state = 0;
        weights = this->array;
    }

    ~Neuron() {}

    const T& operator[](const unsigned& index) const;
    T& operator[](const unsigned& index);

    //Neuron<T>& operator=(const Neuron<T>& neuron);
    Neuron<T>& operator=(const Vector<T>& neuron);
    Neuron<T>& operator()(const Neuron<T>& neuron);

private:
    T state;
    T* weights;
};

template<typename T>
const T& Neuron<T>::operator[](const unsigned& index) const {
    return weights[index];
}

template<typename T>
T& Neuron<T>::operator[](const unsigned& index) {
    return weights[index];
}

/*
template<typename T>
Neuron<T>& Neuron<T>::operator=(const Neuron<T>& neuron) {

    if (this == &neuron) return *this;

    if (numOfInputs == neuron.numOfInputs) {
        state = neuron.state;

        for (unsigned i = 0; i < numOfInputs; ++i)
        {
            this[i] = neuron[i];
        }
    }
    else {
        if (Vector<T>::array != nullptr) {
            delete[] Vector<T>::array;
        }
        numOfInputs = neuron.numOfInputs;
        Vector<T>::len = numOfInputs;
        state = 0;

        Vector<T>::array = new T[Vector<T>::len]();
        weights = Vector<T>::array;

        for (unsigned i = 0; i < Vector<T>::len; ++i)
        {
            this[i] = neuron[i];
        }
    }

    return *this;
}
*/

template<typename T>
Neuron<T>& Neuron<T>::operator=(const Vector<T>& vector) {

    //if (Vector<T>:: == &vector) return *this;

    if (this->getLen() == vector.getLen()) {

        for (unsigned i = 0; i < this->getLen(); ++i)
        {
            this->array[i] = vector[i];
        }

    }
    else {
        if (Vector<T>::array != nullptr) {
            delete[] Vector<T>::array;
        }

        numOfInputs = vector.len;
        Vector<T>::len = numOfInputs;
        state = 0;

        Vector<T>::array = new T[Vector<T>::len]();
        weights = Vector<T>::array;

        for (unsigned i = 0; i < Vector<T>::len; ++i)
        {
            this->array[i] = vector[i];
        }
    }

    return *this;
}

template<typename T>
Neuron<T>& Neuron<T>::operator()(const Neuron<T>& neuron) {
    if (this->array != nullptr) {
        delete[] this->array;
    }
    numOfInputs = neuron.numOfInputs;
    this->len = numOfInputs;
    state = neuron.state;

    this->array = new T[this->len]();
    weights = this->array;

    for (unsigned i = 0; i < this->len; ++i)
    {
        this->array[i] = neuron[i];
    }

    return *this;
}


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
        transferFunction(transferFunction) {

        Matrix<T> w(rows, cols);
        weights(w);

        Neuron<T> o(cols);
        outputs(o);

        Neuron<T> b(cols);
        biases(b);
        
        //pre_deltas_weights(rows, cols),
        bias = static_cast<T>(1);
        
        std::default_random_engine generator;
        std::normal_distribution<T> distribution(0.0, 1);

        for (unsigned i = 0, j; i < cols; ++i) {
            for (j = 0; j < rows; ++j) {
                //weights[i][j] = static_cast<unsigned>(rand() % 2) ? static_cast<T>(rand()) / RAND_MAX : static_cast<T>(rand()) / -RAND_MAX;
                //weights[i][j] = static_cast<T>(rand()) / RAND_MAX;
                //pre_deltas_weights[i][j] = static_cast<T>(0);
                weights[i][j] = distribution(generator);
            }
        }

        for (unsigned i = 0; i < cols; ++i) {
            biases[i] = static_cast<unsigned>(rand() % 2) ? static_cast<T>(rand()) / RAND_MAX : static_cast<T>(rand()) / -RAND_MAX;
            //biases[i] = static_cast<T>(rand()) / RAND_MAX;
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

    Vector<T> BackPropagation(const Vector<T>& input, const Vector<T>& inputs, const T& learning_rate);
    void FeedForward(const Vector<T>&);

private:
    const unsigned cols;
    const unsigned rows;
    Matrix<T> weights;
    Matrix<T> pre_deltas_weights;
    Neuron<T> biases;

    //const 
    void activation_mapper();
};

template<typename T>
Vector<T> Layer<T>::BackPropagation(const Vector<T>& errors, const Vector<T>& input, const T& learning_rate) {
    T bias_Delta = 0;
    //Vector<T> derOfSigmoid(outputs.len);
    Vector<T> gamma(outputs.len);

    //derOfSigmoid = outputs - outputs*outputs;
    switch (transferFunction) {
    case Activation::SIGMOID:
        gamma = errors * (outputs - outputs * outputs);
        break;
    case Activation::SOFTMAX:
        break;
    case Activation::RELU:
        gamma = errors * outputs.reluDerivativeFunc();
        break;
    case Activation::PLAIN:
        gamma = errors * outputs;
        break;
    }

    Matrix<T> in(input.len, 1);
    Matrix<T> wdl(1, gamma.len);
    //    Matrix<T> gradients(in.rows, wdl.cols);

    in[0] = input;
    for (unsigned i = 0; i < gamma.len; ++i) wdl[i][0] = gamma[i];
    for (unsigned i = 0; i < gamma.len; ++i) {
        //bias_Delta *= gamma[i];
        biases[i] -= gamma[i] * learning_rate;
    }
    //bias -= bias_Delta * DEFAULT_BIASLEARNINGRATE;
    //bias -= bias_Delta * learning_rate;

    gamma = weights * gamma; //calculating a pregamma for a next layer

    //weights -= pre_deltas_weights*DEFAULT_MOMENTUM - in * wdl * -learning_rate*(1-DEFAULT_MOMENTUM);
    weights -= in * wdl * learning_rate;

    //pre_deltas_weights = in * wdl * learning_rate;
    return gamma;
}

template<typename T>
void Layer<T>::FeedForward(const Vector<T>& input) {
    outputs = input * weights;
    activation_mapper();
}

template<typename T>
void Layer<T>::activation_mapper() {
    for (unsigned i = 0; i < outputs.len; ++i) {

        switch (transferFunction) {
        case Activation::SIGMOID:
            outputs[i] = Sigmoid(outputs[i] + biases[i]);
            break;
        case Activation::SOFTMAX:
            break;
        case Activation::RELU:
            outputs[i] = Relu(outputs[i] + biases[i]);
            break;
        case Activation::PLAIN:
            outputs[i] = outputs[i] + biases[i];
            break;
        }

    }
}


template<class U, typename T>
class NeuralNetwork {
public:
    T learning_rate;
    mutable T m_error;
    Vector<T> result;

    using element_type = typename std::remove_reference< decltype(std::declval<U>()) >::type;

    explicit NeuralNetwork() : head(nullptr), tail(nullptr), learning_rate(DEFAULT_LEARNINGRATE) {
        std::cout << "The NeuralNetwork has been created" << this << std::endl;
    }

    explicit NeuralNetwork(const T learning_rate) : head(nullptr), tail(nullptr),
        learning_rate(learning_rate),
        m_error(static_cast<T>(0)),
        result(0),
        pDataSet(nullptr) {
        std::cout << "The NeuralNetwork has been created" << this << std::endl;
    }

    ~NeuralNetwork();

    void pushLayer(U& layerObj);
    void mountDataSet(const DataSet<T>& dataset);
    void Train(const unsigned epochs) const;

private:
    void BackPropagation(const Vector<T>& input, const Vector<T>& label) const;
    void FeedForward(const Vector<T>& input) const;
    void getResult() const;
    const void predict(const Vector<T>& input) const;

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
        std::cout << "A Domain has been created..." << this << std::endl;
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
void NeuralNetwork<U, T>::BackPropagation(const Vector<T>& input, const Vector<T>& label) const {
    Domain<U>* current = tail;
    Vector<T> errors(label);
    const Vector<T>* pInput;
    U* pLayer;
    //U* pPreviousLayer;

    pLayer = &current->layer;
    while (current != nullptr) {

        pLayer = &current->layer;
        if (current->pNextDomain == nullptr) {
            errors = pLayer->outputs - label;
        }
        if (current->pPreviousDomain != nullptr) {
            /*
            pPreviousLayer = &current->pPreviousDomain->layer;
            pInput = &pPreviousLayer->outputs;
            */
            pInput = &(&current->pPreviousDomain->layer)->outputs;
        }
        else {
            pInput = &input;
        }

        errors = pLayer->BackPropagation(errors, *pInput, learning_rate);
        //errors = pLayer->weights * errors;
        current = current->pPreviousDomain;
    }
};

template<class U, typename T>
void NeuralNetwork<U, T>::FeedForward(const Vector<T>& input) const {
    Domain<U>* current = head;
    const Vector<T>* pInput;
    pInput = &input;

    U* pLayer;
    while (current != nullptr) {
        pLayer = &current->layer;
        pLayer->FeedForward(*pInput);
        pInput = &pLayer->outputs;
        current = current->pNextDomain;
    }
}

template<class U, typename T>
void NeuralNetwork<U, T>::mountDataSet(const DataSet<T>& dataset) {
    pDataSet = &dataset;
};

template<class U, typename T>
void NeuralNetwork<U, T>::Train(const unsigned epochs) const {
    Domain<U>* current = tail;
    U* pLayer = &current->layer;

    T r;
    T delta_err;
    Vector<T> input(pDataSet->inputs.cols);
    Vector<T> label(pDataSet->labels.rows);

    for (unsigned epoch = 0; epoch < epochs; ++epoch) {
        //std::cout << "!!!!!!!!!!!!!!!!!epoch is: " << epoch << std::endl;

        for (unsigned j = 0; j < pDataSet->inputs.rows; ++j)
        {
            for (unsigned i = 0; i < pDataSet->inputs.cols; ++i) {
                input[i] = pDataSet->inputs[i][j];
            }

            for (unsigned i = 0; i < pDataSet->labels.rows; ++i) {
                label[i] = pDataSet->labels[j][i];
            }

            FeedForward(input);

            m_error = static_cast<T>(0);

            for (unsigned i = 0; i < (pLayer->outputs).len; ++i) {
                delta_err = label[i] - pLayer->outputs[i];
                m_error += delta_err * delta_err;
            }

            r = ((pLayer->outputs).len > 1) ? (pLayer->outputs).len - 1 : 1;
            m_error = m_error / r;
            m_error = static_cast<T>(sqrt(m_error));
            //printf("error = %f\r\n", m_error);

            BackPropagation(input, label);
        }
    }


    //Test
    for (unsigned j = 0; j < pDataSet->inputs.rows; ++j)
    {
        for (unsigned i = 0; i < pDataSet->inputs.cols; ++i) {
            input[i] = pDataSet->inputs[i][j];
        }

        for (unsigned i = 0; i < pDataSet->labels.rows; ++i) {
            label[i] = pDataSet->labels[j][i];
        }
        FeedForward(input);

        for (unsigned i = 0; i < (pLayer->outputs).len; ++i) {
            std::cout << "Target is: " << label[i] << " Output is: " << pLayer->outputs[i] << std::endl;
        }

    }
};

template <typename U, typename T>
void NeuralNetwork<U, T>::getResult() const {
    U* pLayer = tail;
    result = pLayer->outputs;
}

template <typename U, typename T>
const void NeuralNetwork<U, T>::predict(const Vector<T>& input) const {
    FeedForward(input);
}


template <typename T>
struct DataSet {
    DataSet(const Matrix<T>& inputs, const Matrix<T>& labels) : inputs(inputs), labels(labels) {};
    const Matrix<T>& inputs;
    const Matrix<T>& labels;
};


int main()
{
    /*
        const Matrix<double> m(3, 3);
        Vector<double> v(3);

        m[0][0] = -3; m[1][0] = 4; m[2][0] = 1;
        m[0][1] = 7; m[1][1] = -6; m[2][1] = 8;
        m[0][2] = 9; m[1][2] = 3; m[2][2] = -8;

        v[0] = 3; v[1] = 1; v[2] = 7;

        Vector<double> res(3);

        res = m*v;

        for (unsigned i = 0; i < 3; ++i) {
            std::cout << "Result " << res[i] << std::endl;
        }

        const Matrix<double> m1(3, 1);
        const Matrix<double> m2(1, 3);

        const Matrix<double> m3(3, 3);
        const Matrix<double> r(3, 3);


        m1[0][0] = 1; m1[0][1] = 2; m1[0][2] = 3;
        m2[0][0] = 1; m2[1][0] = 2; m2[2][0] = 3;

        r = m1 * m2;

        for (unsigned j = 0, i; j < 3; ++j) {
            for (i = 0; i < 3; ++i) {
                std::cout << r[i][j] << "   ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        m3[0][0] = 1; m3[1][0] = 1; m3[2][0] = 1;
        m3[0][1] = 1; m3[1][1] = 1; m3[2][1] = 1;
        m3[0][2] = 1; m3[1][2] = 1; m3[2][2] = 1;


        r = r - m3;

        for (unsigned j = 0, i; j < 3; ++j) {
            for (i = 0; i < 3; ++i) {
                std::cout << r[i][j] << "   ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        r = r * -0.15;

        for (unsigned j = 0, i; j < 3; ++j) {
            for (i = 0; i < 3; ++i) {
                std::cout << r[i][j] << "   ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    */
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


    srand(time(NULL));

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
    
    NeuralNetwork<Layer<double>, double> NeuralNetwork(0.25);
    NeuralNetwork.pushLayer(layer1);
    NeuralNetwork.pushLayer(layer2);
    NeuralNetwork.pushLayer(layer3);
    NeuralNetwork.pushLayer(layer4);
    
    DataSet<double> dataset(inputs, expectedLabels);
    NeuralNetwork.mountDataSet(dataset);
    NeuralNetwork.Train(8000);


   //NeuralNetwork.pushLayer(layer0);
   //NeuralNetwork.pushLayer(Layer<double>(120, 64));
   //NeuralNetwork.pushLayer(Layer<double>(64, 32));
   //NeuralNetwork.pushLayer(Layer<double>(32, 16));
   //NeuralNetwork.pushLayer(Layer<double>(16, 10));


    std::cout << "Hello World!\n";

    return 0;
}
