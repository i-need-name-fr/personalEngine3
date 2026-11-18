#pragma once

#include "../vulkanContext/moxContext2.hpp"

namespace mox
{
namespace math
{
    enum moxMathError
    {
        moxMathErrorVectorDifferentSize,
        moxMathErrorVectorOutofRange
    };

    template<typename Type>
    requires std::is_arithmetic_v<Type>
    class mVector final
    {
    public:

        mVector() noexcept
        {
            // when no variables provided , we automatically set it to 1 length , and with 0 
            __data.resize(1);
            __data[0] = 0;
        }

        mVector(const std::vector<Type> &var) noexcept
        {
            __data = var;
        }

        mVector(const std::initializer_list<Type> &var) noexcept
        {
            __data = var;
        }

        mVector(std::vector<Type> &var) noexcept
        {
            __data = var;
        }

        mVector(std::initializer_list<Type> &var) noexcept
        {
            __data = var;
        }
        
        mVector(std::vector<Type> &&var) noexcept
        {
            __data = std::move(var);
        }

        mVector(std::initializer_list<Type> &&var) noexcept
        {
            __data = std::move(var);
        }

        mVector(const mVector<Type>& input) noexcept
        {
            if(this == &input) return;
            __data = input.__data;
        }

        mVector(mVector<Type>& input) noexcept
        {
            if(this == &input) return;
            __data = input.__data;
        }

        mVector& operator=(const mVector<Type>& input) noexcept
        {
            if(this == &input) return *this;
            __data = input.__data;
            return *this;
        }

        mVector& operator=(mVector<Type>& input) noexcept
        {
            if(this == &input) return *this;
            __data = input.__data;
            return *this;
        }

        std::uint64_t size() const noexcept
        {
            return __data.size();
        }

        bool operator==(const mVector<Type>& input) const noexcept 
        {
            if(this == &input) return true;
            if(input.size() != this->size()) return false;

            const std::uint64_t size = this->size();
            for(uint32_t i = 0 ; i < size ; i++)
            {
                if(__data[i] != input.__data[i]) return false;
            }
            return true;
        }

        mVector operator+(const mVector<Type>& input) const
        {
            try
            {
                return _implOperatorType(input , moxOperatorPlus);
            }
            catch(std::exception &e)
            {
                throw std::runtime_error(std::format("failed to process operator , Error: {}" , e.what()));
            }
            return *this;
        }

        mVector operator-(const mVector<Type>& input) const
        {
            try
            {
                return _implOperatorType(input , moxOperatorMinus);
            }
            catch(std::exception &e)
            {
                throw std::runtime_error(std::format("failed to process operator , Error: {}" , e.what()));
            }
            return *this;
        }

        mVector& operator+=(const mVector<Type>& input) 
        {
            try
            {
                *this = _implOperatorType(input , moxOperatorPlus);
            }
            catch(std::exception &e)
            {
                throw std::runtime_error(std::format("failed to process operator , Error: {}" , e.what()));
            }
            return *this;
        }

        mVector& operator-=(const mVector<Type>& input) 
        {
            try
            {
                *this = _implOperatorType(input , moxOperatorMinus);
            }
            catch(std::exception &e)
            {
                throw std::runtime_error(std::format("failed to process operator , Error: {}" , e.what()));
            }
            return *this;
        }

        mVector operator*(const mVector<Type>& input) const
        {
            try
            {
                return _implOperatorType(input , moxOperatorMul);
            }
            catch(std::exception &e)
            {
                throw std::runtime_error(std::format("failed to process operator , Error: {}" , e.what()));
            }
        }

        mVector operator/(const mVector<Type>& input) const
        {
            try
            {
                return _implOperatorType(input , moxOperatorDiv);
            }
            catch(std::exception &e)
            {
                throw std::runtime_error(std::format("failed to process operator , Error: {}" , e.what()));
            }
        }

        mVector& operator*=(const mVector<Type>& input) 
        {
            try
            {
                *this = _implOperatorType(input , moxOperatorMul);
            }
            catch(std::exception &e)
            {
                throw std::runtime_error(std::format("failed to process operator , Error: {}" , e.what()));
            }
            return *this;
        }

        mVector& operator/=(const mVector<Type>& input) 
        {
            try
            {
                *this = _implOperatorType(input , moxOperatorDiv);
            }
            catch(std::exception &e)
            {
                throw std::runtime_error(std::format("failed to process operator , Error: {}" , e.what()));
            }
            return *this;
        }

        mVector operator*(const Type scalar) const
        {
            mVector<Type> result = *this;
            for(auto& x : result.__data) x *= scalar;
            return result;
        }

        mVector operator/(const Type scalar) const
        {
            _checkScalarDivisor(scalar);
            mVector<Type> result = *this;
            for(auto& x : result.__data) x /= scalar;
            return result;
        }

        mVector& operator*=(const Type scalar)
        {
            for(auto& x : __data) x *= scalar;
            return *this;
        }

        mVector& operator/=(const Type scalar)
        {
            _checkScalarDivisor(scalar);
            for(auto& x : __data) x /= scalar;
            return *this;
        }

        ~mVector() = default;
        Type& operator[](const std::uint32_t index)
        {
            std::lock_guard<std::shared_mutex> lock(mutex);
            if(size() <= index)
            {
                throw std::runtime_error(std::format("failed to perform operator for vector , since the index of out of range, range -> {} , index -> {}\n" , size() , index));
            }

            return __data[index];
        }

        const Type& operator[](const std::uint32_t index) const 
        {
            if(size() <= index)
            {
                throw std::runtime_error(std::format("failed to perform operator for vector , since the index of out of range, range -> {} , index -> {}\n" , size() , index));
            }

            return __data[index];
        }

        const Type& at(const std::uint32_t index) const 
        {
            if(size() <= index)
            {
                throw std::runtime_error(std::format("failed to perform operator for vector , since the index of out of range, range -> {} , index -> {}\n" , size() , index));
            }

            return __data[index];
        }

        Type& at(const std::uint32_t index)
        {
            std::lock_guard<std::shared_mutex> lock(mutex);
            if(size() <= index)
            {
                throw std::runtime_error(std::format("failed to perform operator for vector , since the index of out of range, range -> {} , index -> {}\n" , size() , index));
            }

            return __data[index];
        }

        const Type sum() const noexcept {return _implSumVariables();}

        const Type mean() const noexcept
        {
            return sum() / static_cast<Type>(size());
        }

        void append(const Type& __var) noexcept
        {
            __data.push_back(__var);
        }

        void append(Type&& __var) noexcept
        {
            __data.push_back(__var);
        }

        void pop() noexcept
        {
            if(__data.empty()) return;
            __data.pop_back();
        }

        void resize(const std::size_t size) noexcept
        {
            __data.resize(size);
        }

    private:

        enum moxOperators
        {
            moxOperatorPlus,
            moxOperatorMinus,
            moxOperatorMul,
            moxOperatorDiv,
            moxOperatorEqual
        };

        std::vector<Type> __data{};
        std::shared_mutex mutex{};

        static void _checkScalarDivisor(const Type scalar)
        {
            if constexpr (std::is_integral_v<Type>)
            {
                if(scalar == 0) throw std::runtime_error("division of vector by zero");
            }
        }

        const Type _implSumVariables() const noexcept
        {
            if(__data.empty()) return 0;
            Type sum = 0;
            for(const auto& num : __data)
            {
                sum += num;
            }
            return sum;
        }


        mVector<Type> _implOperatorType(const mVector<Type>& var , const moxOperators op) const
        {
            if(__data.empty())
            {
                throw std::runtime_error("couldnt process the operator , since the vector is empty");
            }

            if(var.size() != this->size())
            {
                throw std::runtime_error(std::format("failed to process operator , since the vectors has different types , our -> {} , their -> {}" , size() , var.size()));
            }

            const std::uint64_t size = var.size();
            mVector<Type> vec = *this;
            vec.resize(size);
            for(uint32_t i = 0 ; i < size ; i++)
            {
                if(op == moxOperatorPlus)
                {
                    vec.__data[i] += var.__data[i];
                }
                else if(op == moxOperatorMinus)
                {
                    vec.__data[i] -= var.__data[i];
                }
                else if(op == moxOperatorMul)
                {
                    vec.__data[i] *= var.__data[i];
                }
                else if(op == moxOperatorDiv)
                {
                    vec.__data[i] /= var.__data[i];
                }
                else if(op == moxOperatorEqual)
                {
                    vec.__data[i] = var.__data[i];
                }
            }
            return vec;
        }
    };


    // scalar * vector
    template<typename Type>
    mVector<Type> operator*(const Type scalar , const mVector<Type>& v)
    {
        return v * scalar;
    }

    // dot
    template<typename Type>
    Type mDot(const mVector<Type>& a , const mVector<Type>& b) noexcept
    {

        if(a.size() == 0 || b.size() == 0 ) return 0;

        if(a.size() != b.size())
        {
            std::cerr << "the size of 2 vector is different from each other , performing clamp to lowest" << '\n';
        }

        const std::uint64_t size = std::min(a.size() , b.size());
        Type result = (Type)0;
        for(uint32_t i = 0 ; i < size ; i++)
        {
            result += a[i] * b[i];
        }
        return result;
    }

    // length
    template<typename Type>
    const float mLength(const mVector<Type>& a) noexcept
    {
        float result = 0.0f;
        if(a.size() == 0) return result;
        for(uint32_t i = 0 ; i < a.size() ; i++)
        {
            result += a[i] * a[i];
        }

        return std::sqrt(result);
    }
    // normalize
    template<typename Type>
    mVector<Type> mNormalize(const mVector<Type>& a) noexcept
    {
        const float length = math::mLength(a);
        if(length == 0.0f) return mVector<Type>{};

        mVector<Type> result{};
        result.resize(a.size());

        for(uint32_t i = 0 ; i < a.size() ; i++)
        {
            result[i] = a[i] / length;
        }

        return result;
    }
}
}
