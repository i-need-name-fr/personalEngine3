#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <iostream>
#include <fstream>
#include <source_location>
#include <format>
#include <bit>
#include <expected>
#include <string>
#include <string_view>
#include <vector>
#include <sstream>
#include <chrono>
#include <ctime>
#include <mutex>
#include <atomic>
#include <cstdint>
#include <unordered_map>

#define LOG_FILE_OUTPUT "../log.txt"

namespace mox{

    enum class ErrorSignature{
        ErrorTypeCRITICAL = 0lu,
        ErrorTypeWARNING = 1lu,
        ErrorTypeSUCCESS = 2lu,
        ErrorTypeFAILED_CREATION = 3lu,
        ErrorTypeMEDIUM = 4lu,
        ErrorTypeFAILED = 5lu,
        ErrorTypeNONE = 3454lu
    };

    // how many times a message that can repeat every frame is written ( the callers count their messages with it )
    inline static constexpr uint32_t totalMessageCount = 10ul;

    const ErrorSignature MOX_ERROR_TYPE_CRITICAL = ErrorSignature::ErrorTypeCRITICAL;
    const ErrorSignature MOX_ERROR_TYPE_FAILED = ErrorSignature::ErrorTypeFAILED;
    const ErrorSignature MOX_ERROR_TYPE_NONE = ErrorSignature::ErrorTypeNONE;
    const ErrorSignature MOX_ERROR_TYPE_MEDIUM = ErrorSignature::ErrorTypeMEDIUM;
    const ErrorSignature MOX_ERROR_TYPE_FAILED_CREATION = ErrorSignature::ErrorTypeFAILED_CREATION;
    const ErrorSignature MOX_ERROR_TYPE_WARNING = ErrorSignature::ErrorTypeWARNING;
    const ErrorSignature MOX_ERROR_TYPE_SUCCESS = ErrorSignature::ErrorTypeSUCCESS;


    using ErrorDataType = std::pair<ErrorSignature , std::string>;
    template<typename T>
    using ErrorDataOutput = std::expected<T , ErrorDataType>;

    // A message is a page : the number , the time , the type , the place and the text . the number is in the first and in the last line ,
    // so it is easy to see where a message starts and where it ends , and to find it ( "MESSAGE #0043" ) .
    // [[MOX_LOG]] [[<time>]] [[<error type>]] 
    // the whole page is built first and written with ONE write under a mutex , so the pages of different threads never mix
    struct EngineLogger_T{
    private:

        inline static constexpr size_t pageWidth = 80;
        inline static constexpr size_t textWidth = 66;

        std::mutex mutex{};
        std::ofstream defaultFile{};
        // every file has its own numbers : the pages of another file do not leave a gap in the numbers of the log
        std::unordered_map<std::string , uint32_t> messageIds{};
        std::atomic<uint32_t> frame{UINT32_MAX};

        [[nodiscard]] const char* _implGetErrorSignature(const ErrorSignature type) const noexcept{
            switch(type){
                case ErrorSignature::ErrorTypeWARNING : return "WARNING";
                case ErrorSignature::ErrorTypeMEDIUM : return "MEDIUM";
                case ErrorSignature::ErrorTypeCRITICAL : return "CRITICAL";
                case ErrorSignature::ErrorTypeFAILED : return "FAILED";
                case ErrorSignature::ErrorTypeFAILED_CREATION : return "FAILED_CREATION";
                case ErrorSignature::ErrorTypeSUCCESS : return "SUCCESS";
                default : return "NONE";
            }
        }


        // the page of one message ( the number is given by the caller : it is taken under the mutex , so the numbers go in the order of the file )
        [[nodiscard]] std::string _implBuildPage(const uint32_t id , const ErrorSignature type , const std::string& message) const{

            const auto errorType = _implGetErrorSignature(type);
            const auto now   = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
            const auto local = std::chrono::zoned_time{std::chrono::current_zone(), now};
            const std::string data = std::format("[[MOX_LOG]] [[{:%H:%M:%S}]] [[{}]] {}\n" , local , errorType , message);

            return data;
        }   

        [[gnu::hot]] void _implSetupMessage(const mox::ErrorSignature errorType , const std::string& message , const std::string& fileOutput = LOG_FILE_OUTPUT){
            // one lock for the number , the page and the write : the pages are in the file in the order of their numbers
            std::lock_guard<std::mutex> lock(mutex);

            const std::string page = _implBuildPage(++messageIds[fileOutput] , errorType , message);

            if(fileOutput == LOG_FILE_OUTPUT){
                if(!defaultFile.is_open()){
                    std::cerr << std::format("WARNING : couldnt find the file for the logging. \nGiven path is {}\n" , fileOutput);
                    return;
                }
                // flush : the page is in the file even if the program crashes right after it
                defaultFile.write(page.data() , static_cast<std::streamsize>(page.size()));
                defaultFile.flush();
                return;
            }

            // another file was asked for
            std::ofstream file(fileOutput , std::ios::app);
            if(!file.is_open()){
                std::cerr << std::format("WARNING : couldnt find the file for the logging. \nGiven path is {}\n" , fileOutput);
                return;
            }
            file.write(page.data() , static_cast<std::streamsize>(page.size()));
        }

    public:

        EngineLogger_T(const std::string path = LOG_FILE_OUTPUT){
            // a new run starts a new log
            defaultFile.open(path , std::ios::out | std::ios::trunc);
        }

        EngineLogger_T(const EngineLogger_T&) = delete;
        EngineLogger_T& operator=(const EngineLogger_T&) = delete;

        struct messageCreateInfo{
            mox::ErrorSignature errorType;
            std::string message;
        };

        // the number of the current frame ( it is written in every message ) : the frame loop gives it , until then the messages say "-"
        void setFrame(const uint32_t frameIndex) noexcept{
            frame.store(frameIndex , std::memory_order_relaxed);
        }

        void operator()(const void* info , const std::string &path = LOG_FILE_OUTPUT){
            if(!info) return;
            const messageCreateInfo* data = static_cast<const messageCreateInfo*>(info);
            _implSetupMessage(data->errorType , data->message , path);
        }

        void operator()(const ErrorDataType *info, const std::string &path = LOG_FILE_OUTPUT){
            if(!info) return;
            _implSetupMessage(info->first , info->second , path);
        }

        void operator()(const ErrorDataType &info, const std::string &path = LOG_FILE_OUTPUT){
            _implSetupMessage(info.first , info.second , path);
        }

        void operator()(const ErrorDataOutput<void> &info, const std::string &path = LOG_FILE_OUTPUT){
            if(info.has_value()) return;
            _implSetupMessage(info.error().first , info.error().second , path);
        }

        template<typename T>
        void operator()(const ErrorDataOutput<T> __i) noexcept
        {
            if(__i.has_value()) return;
            _implSetupMessage(__i.error().first , __i.error().second , LOG_FILE_OUTPUT );
        }

        void operator()(const std::string message) noexcept
        {
            _implSetupMessage(MOX_ERROR_TYPE_NONE , message);
        }
    };

    inline EngineLogger_T engineLogger{};

}

#define MOX_CHECK_RESULT(result) if(!result.has_value()) return std::unexpected(result.error());

#endif
