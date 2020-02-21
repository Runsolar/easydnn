//Under construction, it doesn't work yet
/*
This code is devoted to my father Jesus Christ.
Code by St. Spirit and Danijar Wolf, Feb 20, 2020.
*/

#include <iostream>
#include<ctime>

template<typename T> class Matrix;

template<typename T>
float Sigmoid(const T& x)
{
    return 1 / (1 + exp(-x));
}

template<typename T>
class Vector {
public:
    using type = T;

    Vector() = delete;
    explicit Vector(const int len) : len(len) {
        try {
            array = new T[len]();
            std::cout << "A new vector hase been created... " << this << std::endl;
        }
        catch (std::exception & ex) {
            std::cout << "En exception is happened... " << ex.what() << std::endl;
            return;
        }
    }

    Vector(const Vector<T>& vec): Vector(vec.len) {
        std::cout << "Coppy... this: " << this <<  std::endl;
        if (this == &vec) return;
        for (int i = 0; i < len; ++i) {
            array[i] = vec.array[i];
        }
    }

    ~Vector() {
        delete[] array;
        std::cout << "A vector hase been deleted... " << this << std::endl;
    }

    //Vector<T> operator+(const Vector<T>& vector);
    //T operator*(const Vector<T>& vector);

    T operator*(const Vector<T>& vector) const;
    Vector<T> operator*(const Matrix<T>& matrix) const;

    //Vector<T> operator*(const T& scalar);
    

    //Vector<T>& operator*=(const T& scalar);
    //Vector<T>& operator+=(const Vector<T>& vector);
    //Vector<T>& operator-=(const Vector<T>& vector);

    Vector<T>& operator=(const Vector<T>& vector);

    T& operator[](const int index) const {
        return array[index];
    }

    
private:
    int len;
    T* array;
};

/*
template<typename T>
Vector<T> Vector<T>::operator+(const Vector<T>& vector) {
    if (this->len == vector->len) {
        for (int i = 0; i < this->len; ++i) {
            this->array[i] = this->array[i] + vector->array[i];
        }
    }
    return *this;
}

template<typename T>
T Vector<T>::operator*(const Vector<T>& vector) {
    T sum = 0;
    if (this->len == vector.len) {
        for (int i = 0; i < this->len; ++i) {
            sum += this->array[i] * vector.array[i];
        }
    }
    return sum;
}
*/

template<typename T>
T Vector<T>::operator*(const Vector<T>& vector) const {
    T sum = 0;
    if (this->len == vector.len) {
        for (int i = 0; i < this->len; ++i) {
            sum += this->array[i] * vector.array[i];
        }
    }
    return sum;
}

template<typename T>
Vector<T> Vector<T>::operator*(const Matrix<T>& matrix) const {
    Vector<T> vec(matrix.cols);
    for (int i = 0; i < matrix.cols; i++) {
        vec[i] = *this * matrix[i];
    }
    return vec;
}

/*
template<typename T>
Vector<T> Vector<T>::operator*(const T& scalar) {
    Vector<T> vec(this->len);

    for (int i = 0; i < this->len; ++i) {
        vec[i] = this->array[i] * scalar;
    }
    return vec;
}
*/


/*
template<typename T>
Vector<T>& Vector<T>::operator*=(const T& scalar) {
    for (int i = 0; i < this->len; ++i) {
        this->array[i] *= scalar;
    }
    return *this;
}
*/

/*
template<typename T>
Vector<T>& Vector<T>::operator+=(const Vector<T>& vector) {
    if (this->len == vector->len) {
        for (int i = 0; i < this->len; ++i) {
            this->array[i] += vector->array[i];
        }
    }
    return *this;
}
*/
/*
template<typename T>
Vector<T>& Vector<T>::operator-=(const Vector<T>& vector) {
    this += vector * (-1);
    return *this;
}
*/
template<typename T>
Vector<T>& Vector<T>::operator=(const Vector<T>& vector) {
    if (this == &vector) return *this;
    if (this->len == vector.len) {
        for (int i = 0; i < this->len; ++i) {
            array[i] = vector.array[i];
        }
    }
    else {
        this->len = vector.len;
        delete[] array;
        array = new T[this->len]();
        for (int i = 0; i < this->len; ++i) {
            array[i] = vector.array[i];
        }
    }
    return *this;
}

template<typename T>
class Matrix {
    friend Vector<T>;

public:
    Matrix() = delete;

    explicit Matrix(const int rows, const int cols) : rows(rows), cols(cols) {
        matrix = new Vector<T> * [cols];
        for (int i = 0; i < cols; i++) {
            matrix[i] = new Vector<T>(rows);
        }

        std::cout << "A new matrix has been created... " << this << std::endl;
    }

    Vector<T>& operator[](const int index) const {
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
    Vector<T>** matrix;
};

template<typename T>
class Layer {

public:
    Vector<T> outputs;
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
                //Vector<T> vector = weights[i];
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

    void FeedForward(Vector<T>&);

private:
    const int cols;
    const int rows;
    Matrix<T> weights;  
};

template<typename T>
void Layer<T>::FeedForward(Vector<T>& input) {
    outputs = input * weights;
}

template<class U>
class NeuralNetwork {
public:
    NeuralNetwork() : head(nullptr), tail(nullptr) {
        std::cout << "The NeuralNetwork has been created" << this << std::endl;
    }
    ~NeuralNetwork();

    void pushLayer(U& layerObj);
    void loadDataSet(const Matrix<double>& inputs, const Vector<double>& labels) const;
    void Train();
    

private:
    void FeedForward(Vector<double>& input);

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

template<class U>
NeuralNetwork<U>::~NeuralNetwork() {
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

template<class U>
void NeuralNetwork<U>::pushLayer(U& layerObj) {
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

template<class U>
void NeuralNetwork<U>::FeedForward(Vector<double>& input) {
    Domain<U>* current = head;
    Vector<double>* pInput;

    pInput = &input;

    while (current != nullptr) {
        U* layer = &current->layer;
        layer->FeedForward(*pInput);
        pInput = &layer->outputs;
        current = current->pNextDomain;
        //layer->PrintOutputs(); 
        //std::cout << std::endl;
    }
}

template<class U>
void NeuralNetwork<U>::loadDataSet(const Matrix<double>& inputs, const Vector<double> & labels) const {
    //FeedForward(input);
};

template<class U>
void NeuralNetwork<U>::Train(){
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


    const Vector<double> expectedLabels(8);
    expectedLabels[0] = 0; expectedLabels[1] = 1; expectedLabels[2] = 1;  expectedLabels[3] = 0;
    expectedLabels[4] = 1; expectedLabels[5] = 0; expectedLabels[6] = 0;  expectedLabels[7] = 1;

    Layer<double> layer1(3, 3);
    Layer<double> layer2(3, 9);
    Layer<double> layer3(9, 9);
    Layer<double> layer4(9, 1);

    NeuralNetwork<Layer<double>> NeuralNetwork;
    NeuralNetwork.pushLayer(layer1);
    NeuralNetwork.pushLayer(layer2);
    NeuralNetwork.pushLayer(layer3);
    NeuralNetwork.pushLayer(layer4);

/*
    Vector<double> input(3);
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
