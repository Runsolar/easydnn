//under construction, not work yet

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
    /*
        Vector(const Vector<T>& vecObj) {

        }
    */
    Vector() = delete;
    Vector(const int len) : len(len) {
        try {
            array = new T[len]();
            //std::cout << "A new vector hase been created... " << this << std::endl;
        }
        catch (std::exception & ex) {
            std::cout << "En exception is happened... " << ex.what() << std::endl;
            return;
        }
    }

    ~Vector() {
        delete[] array;
        //std::cout << "A vector hase been deleted... " << this << std::endl;
    }

    Vector<T> operator+(const Vector<T>& vector);
    T operator*(const Vector<T>& vector);
    Vector<T> operator*(const T& scalar);
    Vector<T> operator*(const Matrix<T>& matrix);

    Vector<T>& operator*=(const T& scalar);
    Vector<T>& operator+=(const Vector<T>& vector);
    Vector<T>& operator-=(const Vector<T>& vector);

    Vector<T>& operator=(const Vector<T>& vector);

    T& operator[](const int index) {
        return array[index];
    }

private:
    const int len;
    T* array;
};

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
    if (this->len == vector->len) {
        for (int i = 0; i < this->len; ++i) {
            sum += this->array[i] * vector->array[i];
        }
    }
    return sum;
}

template<typename T>
Vector<T> Vector<T>::operator*(const T& scalar) {
    Vector<T> vec(this->len);

    for (int i = 0; i < this->len; ++i) {
        vec[i] = this->array[i] * scalar;
    }
    return vec;
}

template<typename T>
Vector<T> Vector<T>::operator*(const Matrix<T>& matrix) {
    Vector<T> vec(matrix.cols);
    for (int i = 0; i < matrix.cols; i++) {
        vec[i] = this * matrix[i];
    }
    return vec;
}

template<typename T>
Vector<T>& Vector<T>::operator*=(const T& scalar) {
    for (int i = 0; i < this->len; ++i) {
        this->array[i] *= scalar;
    }
    return *this;
}

template<typename T>
Vector<T>& Vector<T>::operator+=(const Vector<T>& vector) {
    if (this->len == vector->len) {
        for (int i = 0; i < this->len; ++i) {
            this->array[i] += vector->array[i];
        }
    }
    return *this;
}

template<typename T>
Vector<T>& Vector<T>::operator-=(const Vector<T>& vector) {
    this += vector * (-1);
    return *this;
}

template<typename T>
Vector<T>& Vector<T>::operator=(const Vector<T>& vector) {
    if (vector == this) return *this;
    if (this->len == vector->len) {
        for (int i = 0; i < this->len; ++i) {
            this[i] = vector[i];
        }
    }
    return *this;
}

template<typename T>
class Matrix {
public:
    Matrix(const int rows, const int cols) : rows(rows), cols(cols) {
        matrix = new Vector<T> * [cols];
        for (int i = 0; i < cols; i++) {
            matrix[i] = new Vector<T>(rows);
        }

        std::cout << "A new matrix has been created... " << this << std::endl;
    }

    Vector<T>& operator[](const int index) {
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

    void PrintLayer() {
        for (int i = 0, j; i < cols; ++i) {
            for (j = 0; j < rows; ++j) {
                std::cout << static_cast<T>(weights[i][j]) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    void FeedForward(const Vector<T>&);

    ~Layer() {}
private:
    const int cols;
    const int rows;
    Matrix<T> weights;
    Vector<T> outputs;
    T* bias;    // bias of this layer
};

template<typename T>
void Layer<T>::FeedForward(const Vector<T>& input) {
    outputs = input * weights;
}

template<typename T>
class NeuralNetwork {
public:
    NeuralNetwork() : head(nullptr) {
        std::cout << "The NeuralNetwork has been created" << this << std::endl;
    }
    ~NeuralNetwork();

    void pushLayer(const T& layerObj);
    void FeedForward();

private:
    template<typename T>
    class Domain {
    public:
        const T& layer;
        Domain<T>* pNextDomain;
        Domain<T>* pPreviousDomain;
        Domain(const T& layer = T(), Domain<T>* pNextDomain = nullptr, Domain<T>* pPreviousDomain = nullptr) :
            layer(layer),
            pNextDomain(pNextDomain),
            pPreviousDomain(pPreviousDomain) {}
    };
    Domain<T>* head;
};

template<typename T>
NeuralNetwork<T>::~NeuralNetwork() {
    if (head != nullptr) {
        while (head != nullptr) {
            Domain<T>* current = head;
            head = current->pNextDomain;
            delete current;
            std::cout << "A Domain has been deleted... " << this << std::endl;
        }
        std::cout << "The NeuralNetwork has been deleted... " << this << std::endl;
    }
}

template<typename T>
void NeuralNetwork<T>::pushLayer(const T& layerObj) {
    if (head == nullptr) {
        head = new Domain<T>(layerObj);
        std::cout << "A first Domain has been created..." << this << std::endl;
        return;
    }

    Domain<T>* current = head;
    while (current->pNextDomain != nullptr) {
        current = current->pNextDomain;
    }
    current->pNextDomain = new Domain<T>(layerObj);
    std::cout << "A Domain has been created..." << this << std::endl;
    return;
}

template<typename T>
void NeuralNetwork<T>::FeedForward() {

}


int main()
{
    //Default Inputs
    const float inputs[8][3] = {
      {0, 0, 0}, //0
      {0, 0, 1}, //1
      {0, 1, 0}, //1
      {0, 1, 1}, //0
      {1, 0, 0}, //1
      {1, 0, 1}, //0
      {1, 1, 0}, //0
      {1, 1, 1}  //1
    };

    // values that we were expecting to get from the 4th/(output)layer of Neural-NeuralNetwork, in other words something like a feedback to the Neural-NeuralNetwork.
    const float expectedOutput[8][1] = { {0}, {1}, {1}, {0}, {1}, {0}, {0}, {1} };
    /*
        Layer<double> layer1(120, 64);
        Layer<double> layer2(64, 32);
        Layer<double> layer3(32, 16);
        Layer<double> layer4(16, 10);
    */
    //srand(time(NULL));

    Layer<double> layer1(3, 3);
    layer1.PrintLayer();
    Layer<double> layer2(3, 9);
    layer2.PrintLayer();
    Layer<double> layer3(9, 9);
    layer3.PrintLayer();
    Layer<double> layer4(9, 1);
    layer4.PrintLayer();


    NeuralNetwork<Layer<double>> NeuralNetwork;
    NeuralNetwork.pushLayer(layer1);
    NeuralNetwork.pushLayer(layer2);
    NeuralNetwork.pushLayer(layer3);
    NeuralNetwork.pushLayer(layer4);

    //NeuralNetwork.pushLayer(layer0);
    //NeuralNetwork.pushLayer(Layer<double>(120, 64));
    //NeuralNetwork.pushLayer(Layer<double>(64, 32));
    //NeuralNetwork.pushLayer(Layer<double>(32, 16));
    //NeuralNetwork.pushLayer(Layer<double>(16, 10));


    std::cout << "Hello World!\n";
}
