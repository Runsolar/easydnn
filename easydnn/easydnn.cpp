//Under construction, it doesn't work yet
/*
This code is devoted to my father Jesus Christ.
Code by St. Spirit and Danijar Wolf, Feb 20, 2020.
*/

#include <iostream>
#include<ctime>

template<typename T> class Matrix;
template<class U, typename T> class NeuralNetwork;
template<typename T> class Layer;

template<typename T>
T Sigmoid(const T& x)
{
    return 1 / (1 + exp(-x));
}

template<typename T>
class Neuron {
    friend Layer<T>;
    //friend NeuralNetwork<Layer<T>, T>;

public:
    Neuron() = delete;
    explicit Neuron(const int len) : len(len) {
        try {
            array = new T[len]();
            std::cout << "A new Neuron hase been created... " << this << std::endl;
        }
        catch (std::exception & ex) {
            std::cout << "En exception is happened... " << ex.what() << std::endl;
            return;
        }
    }

    Neuron(const Neuron<T>& vec): Neuron(vec.len) {
        std::cout << "Coppy... this: " << this <<  std::endl;
        if (this == &vec) return;
        for (int i = 0; i < len; ++i) {
            array[i] = vec.array[i];
        }
    }

    ~Neuron() {
        delete[] array;
        std::cout << "A Neuron hase been deleted... " << this << std::endl;
    }

    //Neuron<T> operator+(const Neuron<T>& Neuron);
    //T operator*(const Neuron<T>& Neuron);

    T operator*(const Neuron<T>& neuron) const;
    Neuron<T> operator*(const Matrix<T>& matrix) const;
    
    Neuron<T>& operator-=(const Neuron<T>& neuron);
    Neuron<T> operator-(const Neuron<T>& neuron);

    //Neuron<T> operator*(const T& scalar);
    

    //Neuron<T>& operator*=(const T& scalar);
    //Neuron<T>& operator+=(const Neuron<T>& neuron);


    Neuron<T>& operator=(const Neuron<T>& neuron);

    T& operator[](const int index) const {
        return array[index];
    }

    
private:
    int len;
    T* array;
};

/*
template<typename T>
Neuron<T> Neuron<T>::operator+(const Neuron<T>& Neuron) {
    if (this->len == Neuron->len) {
        for (int i = 0; i < this->len; ++i) {
            this->array[i] = this->array[i] + Neuron->array[i];
        }
    }
    return *this;
}

template<typename T>
T Neuron<T>::operator*(const Neuron<T>& Neuron) {
    T sum = 0;
    if (this->len == neuron.len) {
        for (int i = 0; i < this->len; ++i) {
            sum += this->array[i] * neuron.array[i];
        }
    }
    return sum;
}
*/

template<typename T>
T Neuron<T>::operator*(const Neuron<T>& neuron) const {
    T sum = 0;
    if (this->len == neuron.len) {
        for (int i = 0; i < this->len; ++i) {
            sum += this->array[i] * neuron.array[i];
        }
    }
    return sum;
}

template<typename T>
Neuron<T> Neuron<T>::operator*(const Matrix<T>& matrix) const {
    Neuron<T> vec(matrix.cols);
    for (int i = 0; i < matrix.cols; i++) {
        vec[i] = *this * matrix[i];
    }
    return vec;
}

template<typename T>
Neuron<T> Neuron<T>::operator-(const Neuron<T>& neuron) {
    Neuron<T> vec(*this);
    vec -= neuron;
    return vec;
}

template<typename T>
Neuron<T>& Neuron<T>::operator-=(const Neuron<T>& neuron) {
    //this += Neuron * (-1);
    if (this->len == neuron.len) {
        for (int i = 0; i < this->len; i++) {
            this->array[i] -= neuron.array[i];
        }
    }
    return *this;
}



/*
template<typename T>
Neuron<T> Neuron<T>::operator*(const T& scalar) {
    Neuron<T> vec(this->len);

    for (int i = 0; i < this->len; ++i) {
        vec[i] = this->array[i] * scalar;
    }
    return vec;
}
*/


/*
template<typename T>
Neuron<T>& Neuron<T>::operator*=(const T& scalar) {
    for (int i = 0; i < this->len; ++i) {
        this->array[i] *= scalar;
    }
    return *this;
}
*/

/*
template<typename T>
Neuron<T>& Neuron<T>::operator+=(const Neuron<T>& Neuron) {
    if (this->len == Neuron->len) {
        for (int i = 0; i < this->len; ++i) {
            this->array[i] += Neuron->array[i];
        }
    }
    return *this;
}
*/



template<typename T>
Neuron<T>& Neuron<T>::operator=(const Neuron<T>& Neuron) {
    if (this == &Neuron) return *this;
    if (this->len == Neuron.len) {
        for (int i = 0; i < this->len; ++i) {
            array[i] = Neuron.array[i];
        }
    }
    else {
        this->len = Neuron.len;
        
        delete[] array;
        array = new T[this->len]();
        for (int i = 0; i < this->len; ++i) {
            array[i] = Neuron.array[i];
        }
    }
    return *this;
}

template<typename T>
class Matrix {
    friend Neuron<T>;
    friend NeuralNetwork<Layer<T>, T>;

public:
    Matrix() = delete;

    explicit Matrix(const int rows, const int cols) : rows(rows), cols(cols) {
        matrix = new Neuron<T> * [cols];
        for (int i = 0; i < cols; i++) {
            matrix[i] = new Neuron<T>(rows);
        }

        std::cout << "A new matrix has been created... " << this << std::endl;
    }

    Neuron<T>& operator[](const int index) const {
        return *matrix[index];
    }

    ~Matrix() {
        for (int i = 0; i < cols; i++) {
            delete matrix[i];
        }
        delete[] matrix;
        std::cout << "A matrix has been deleted... " << this << std::endl;
    }

private:
    const int cols;
    const int rows;
    Neuron<T>** matrix;
};



template<typename T>
class Layer {
    friend NeuralNetwork<Layer<T>, T>;

public:
    using element_type = typename std::remove_reference< decltype(std::declval<T>) >::type;
    //using type = T;
    //type Type = T();

    Neuron<T> outputs;
    T* bias;    // bias of this layer

    enum activation { SOFTMAX, SIGMOID, RELU };

    Layer(const int numsOfWeights,
        const int numsOfPerceptrons) :
        rows(numsOfWeights),
        cols(numsOfPerceptrons),
        weights(rows, cols),
        outputs(numsOfPerceptrons),
        bias(nullptr) {
        bias = new T(1.0);
        for (int i = 0, j; i < cols; ++i) {
            for (j = 0; j < rows; ++j) {
                //Neuron<T> Neuron = weights[i];
                weights[i][j] = static_cast<int>(rand() % 2) ? static_cast<T>(rand()) / RAND_MAX : static_cast<T>(rand()) / -RAND_MAX;
            }
        }
    }

    ~Layer() {}

    void PrintLayer() const {
        for (int i = 0, j; i < cols; ++i) {
            for (j = 0; j < rows; ++j) {
                std::cout << static_cast<T>(weights[i][j]) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    void PrintOutputs() const {
        for (int i = 0; i < outputs.len; i++) {
            std::cout << outputs[i] << std::endl;
        }
    }

    void BackPropagation(const Neuron<T>& input);
    void FeedForward(const Neuron<T>&);

    
private:
    const int cols;
    const int rows;
    Matrix<T> weights;

    void sigmoid_mapper();
};

template<typename T>
void Layer<T>::BackPropagation(const Neuron<T>& input) {

}

template<typename T>
void Layer<T>::FeedForward(const Neuron<T>& input) {
    outputs = input * weights;
    sigmoid_mapper();
}

template<typename T>
void Layer<T>::sigmoid_mapper() {
    for (int i = 0; i < outputs.len; ++i) {
        outputs[i] = Sigmoid(outputs[i]);
    }
}



template<class U, typename T>
class NeuralNetwork {
public:
    using element_type = typename std::remove_reference< decltype(std::declval<U>()) >::type;

    NeuralNetwork() : head(nullptr), tail(nullptr) {
        std::cout << "The NeuralNetwork has been created" << this << std::endl;
    }
    ~NeuralNetwork();

    void pushLayer(U& layerObj);
    void loadDataSet(const Matrix<T>& inputs, const Neuron<T>& labels);
    void Train();

private:
    void BackPropagation(Neuron<T>& labels);
    void FeedForward(Neuron<T>& input);

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
void NeuralNetwork<U, T>::BackPropagation(Neuron<T>& labels) {
    Domain<U>* current = tail;
    //Neuron<T> errors;
    Neuron<T>* pInput;
    pInput = &labels;

    while (current != nullptr) {
        U* layer = &current->layer;
        layer->BackPropagation(*pInput);

        pInput = &layer->outputs;
        current = current->pPreviousDomain;
    }
    //Neuron<T> errors(layer->cols);
    //errors = layer->outputs - labels;



};

template<class U, typename T>
void NeuralNetwork<U, T>::FeedForward(Neuron<T>& input) {
    Domain<U>* current = head;
    Neuron<T>* pInput;

    pInput = &input;
    U* layer;

    while (current != nullptr) {
        layer = &current->layer;
        layer->FeedForward(*pInput);
        pInput = &layer->outputs;
        current = current->pNextDomain;
        
        layer->PrintOutputs(); 
        std::cout << std::endl;
    }
}

template<class U, typename T>
void NeuralNetwork<U, T>::loadDataSet(const Matrix<T>& inputs, const Neuron<T> & labels) {
    Neuron<T> input(inputs.cols);
    Neuron<T> _labels(labels);

    for (int j = 0; j < inputs.rows; ++j)
    {
        for (int i = 0; i < inputs.cols; ++i) {
            //std::cout << inputs[i][j] << " ";
            input[i] = inputs[i][j];
        }
        std::cout << "Row is..........: " << j <<std::endl;
        FeedForward(input);
        BackPropagation(_labels);
    }

    
};

template<class U, typename T>
void NeuralNetwork<U, T>::Train(){
    //FeedForward(input);
};


int main()
{
    /*
    const double inputs[8][3] = {
      {0, 0, 0}, //0
      {0, 0, 1}, //1
      {0, 1, 0}, //1
      {0, 1, 1}, //0
      {1, 0, 0}, //1
      {1, 0, 1}, //0
      {1, 1, 0}, //0
      {1, 1, 1}  //1
    };
    */

 //   const double expectedLabels[8][1] = { {0}, {1}, {1}, {0}, {1}, {0}, {0}, {1} };

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


    const Neuron<double> expectedLabels(8);
    expectedLabels[0] = 0; expectedLabels[1] = 1; expectedLabels[2] = 1;  expectedLabels[3] = 0;
    expectedLabels[4] = 1; expectedLabels[5] = 0; expectedLabels[6] = 0;  expectedLabels[7] = 1;

    Layer<double> layer1(3, 3);
    Layer<double> layer2(3, 9);
    Layer<double> layer3(9, 9);
    Layer<double> layer4(9, 1);

    NeuralNetwork<Layer<double>, double> NeuralNetwork;
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
    NeuralNetwork.loadDataSet(inputs, expectedLabels);

    //NeuralNetwork.pushLayer(layer0);
    //NeuralNetwork.pushLayer(Layer<double>(120, 64));
    //NeuralNetwork.pushLayer(Layer<double>(64, 32));
    //NeuralNetwork.pushLayer(Layer<double>(32, 16));
    //NeuralNetwork.pushLayer(Layer<double>(16, 10));


    std::cout << "Hello World!\n";
}
