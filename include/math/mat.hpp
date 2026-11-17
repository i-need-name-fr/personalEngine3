#pragma once

#include "vec.hpp"

#include <algorithm>
#include <cstddef>
#include <format>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace mox
{
namespace math
{
    template<typename Type>
    requires std::is_arithmetic_v<Type>
    class mMatrix final
    {
    public:

        mMatrix() noexcept = default;

        mMatrix(const std::size_t rows , const std::size_t cols , const Type value = Type{})
            : data_(rows * cols , value) , rows_(rows) , cols_(cols)
        {
        }

        mMatrix(std::initializer_list<std::initializer_list<Type>> init)
            : rows_(init.size())
        {
            cols_ = init.size() == 0 ? 0 : init.begin()->size();
            data_.reserve(rows_ * cols_);

            for(const auto& row : init)
            {
                if(row.size() != cols_)
                {
                    throw std::runtime_error(std::format("failed to create matrix , rows have different length , expected -> {} , got -> {}" , cols_ , row.size()));
                }
                data_.insert(data_.end() , row.begin() , row.end());
            }
        }


        static mMatrix zeros(const std::size_t rows , const std::size_t cols)
        {
            return mMatrix(rows , cols , Type{});
        }

        static mMatrix identity(const std::size_t n)
        {
            mMatrix result(n , n , Type{});
            for(std::size_t i = 0 ; i < n ; i++)
            {
                result.data_[i * n + i] = static_cast<Type>(1);
            }
            return result;
        }

        std::size_t rows() const noexcept { return rows_; }
        std::size_t cols() const noexcept { return cols_; }

        Type& operator()(const std::size_t row , const std::size_t col)
        {
            _implCheckIndex(row , col);
            return data_[row * cols_ + col];
        }

        const Type& operator()(const std::size_t row , const std::size_t col) const
        {
            _implCheckIndex(row , col);
            return data_[row * cols_ + col];
        }

        bool operator==(const mMatrix& input) const noexcept
        {
            return rows_ == input.rows_ && cols_ == input.cols_ && data_ == input.data_;
        }

        mMatrix operator+(const mMatrix& input) const
        {
            mMatrix result(*this);
            result += input;
            return result;
        }

        mMatrix operator-(const mMatrix& input) const
        {
            mMatrix result(*this);
            result -= input;
            return result;
        }

        mMatrix& operator+=(const mMatrix& input)
        {
            _implCheckShape(input);
            for(std::size_t i = 0 ; i < data_.size() ; i++)
            {
                data_[i] += input.data_[i];
            }
            return *this;
        }

        mMatrix& operator-=(const mMatrix& input)
        {
            _implCheckShape(input);
            for(std::size_t i = 0 ; i < data_.size() ; i++)
            {
                data_[i] -= input.data_[i];
            }
            return *this;
        }

        mMatrix operator*(const mMatrix& input) const
        {
            if(cols_ != input.rows_)
            {
                throw std::runtime_error(std::format("failed to multiply matrices , our cols -> {} , their rows -> {}" , cols_ , input.rows_));
            }

            mMatrix result(rows_ , input.cols_ , Type{});
            for(std::size_t i = 0 ; i < rows_ ; i++)
            {
                for(std::size_t k = 0 ; k < cols_ ; k++)
                {
                    const Type left = data_[i * cols_ + k];
                    for(std::size_t j = 0 ; j < input.cols_ ; j++)
                    {
                        result.data_[i * input.cols_ + j] += left * input.data_[k * input.cols_ + j];
                    }
                }
            }
            return result;
        }

        mVector<Type> operator*(const mVector<Type>& vec) const
        {
            if(cols_ != vec.size())
            {
                throw std::runtime_error(std::format("failed to multiply matrix by vector , matrix cols -> {} , vector size -> {}" , cols_ , vec.size()));
            }

            std::vector<Type> result(rows_ , Type{});
            for(std::size_t i = 0 ; i < rows_ ; i++)
            {
                Type sum = Type{};
                for(std::size_t j = 0 ; j < cols_ ; j++)
                {
                    sum += data_[i * cols_ + j] * vec[static_cast<std::uint32_t>(j)];
                }
                result[i] = sum;
            }
            return mVector<Type>(std::move(result));
        }

        mMatrix operator*(const Type scalar) const
        {
            mMatrix result(*this);
            result *= scalar;
            return result;
        }

        mMatrix& operator*=(const Type scalar)
        {
            for(auto& x : data_) x *= scalar;
            return *this;
        }

        mMatrix transpose() const
        {
            mMatrix result(cols_ , rows_ , Type{});
            for(std::size_t i = 0 ; i < rows_ ; i++)
            {
                for(std::size_t j = 0 ; j < cols_ ; j++)
                {
                    result.data_[j * rows_ + i] = data_[i * cols_ + j];
                }
            }
            return result;
        }

        template<typename Function>
        mMatrix apply(Function&& function) const
        {
            mMatrix result(*this);
            for(auto& x : result.data_)
            {
                x = static_cast<Type>(function(x));
            }
            return result;
        }

    private:

        std::vector<Type> data_{};
        std::size_t rows_{0};
        std::size_t cols_{0};

        void _implCheckIndex(const std::size_t row , const std::size_t col) const
        {
            if(row >= rows_ || col >= cols_)
            {
                throw std::runtime_error(std::format("matrix index out of range , size -> {}x{} , index -> ({}, {})" , rows_ , cols_ , row , col));
            }
        }

        void _implCheckShape(const mMatrix& input) const
        {
            if(rows_ != input.rows_ || cols_ != input.cols_)
            {
                throw std::runtime_error(std::format("matrices have different size , our -> {}x{} , their -> {}x{}" , rows_ , cols_ , input.rows_ , input.cols_));
            }
        }
    };

    template<typename Type>
    mMatrix<Type> operator*(const Type scalar , const mMatrix<Type>& m)
    {
        return m * scalar;
    }
}
}
