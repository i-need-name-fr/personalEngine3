#pragma once

#include "Model3.hpp"
#include "VertexHeap.hpp"
#include "Entity.hpp"

namespace mox
{
namespace utils
{

    enum moxBluePrintType
    {
        moxBluePrintEntity,
        moxBluePrintModel3,
        moxBluePrintUndefined

    };

    struct ForgeBluePrint
    {
        Entity::CreateInfo eInfo{};
        Model3::CreateInfo mInfo{};
        std::string name{moxUndefinedBehavior};
        moxBluePrintType type{moxBluePrintUndefined};
    };

    class EntityForge final
    {
    public:

        SetTypicalOperators(EntityForge)
        SetTypicalTransferOperators(EntityForge)

        struct CreateInfo
        {
            moxContext2* context{nullptr};
            Source::TextureHub* txHub{nullptr};
            Source::LightHub* lHub{nullptr};
        };

        explicit EntityForge(const CreateInfo& info)
        {
            inputData = info;
        }

        void addBluePrint(const ForgeBluePrint& print ) noexcept
        {
            std::lock_guard<std::shared_mutex> lock(mutex);
            bluePrints.push_back(print);
        }

        MOX_NO_DISCARD ErrorDataOutput<void> execute() noexcept {return _implExecuteBluePrints();}

        void setMatrixM(const std::string& name , const glm::mat4 matrix , const std::uint32_t id = 0) noexcept
        {
            std::lock_guard<std::shared_mutex> lock(mutex);
            auto it = _pMd3.find(name);
            if(it != _pMd3.end())
            {
                it->second->setMatrix(matrix , id);
            }
        }

        MOX_NO_DISCARD ErrorDataOutput<void> addMatrixM(const std::string& name , const glm::mat4 matrix) noexcept
        {
            std::lock_guard<std::shared_mutex> lock(mutex);
            auto it = _pMd3.find(name);
            if(it != _pMd3.end())
            {
                return it->second->addMatrix(matrix);
            }
            return {};
        }
        
        MOX_NO_DISCARD ErrorDataOutput<void> changeStatusM(const std::string& name , const bool status) noexcept
        {
            std::lock_guard<std::shared_mutex> lock(mutex);
            auto it = _pMd3.find(name);
            if(it != _pMd3.end())
            {
                return it->second->changeStatus(status);
            }
            return {};
        }

        void setMatrixE(const std::string& name , const glm::mat4 matrix , const std::uint32_t id = 0) noexcept
        {
            std::lock_guard<std::shared_mutex> lock(mutex);
            auto it = _pEnt.find(name);
            if(it != _pEnt.end())
            {
                it->second->setMatrix(id,  matrix);
            }
        }

        MOX_NO_DISCARD ErrorDataOutput<void> addMatrixE(const std::string& name , const glm::mat4 matrix) noexcept
        {
            std::lock_guard<std::shared_mutex> lock(mutex);
            auto it = _pEnt.find(name);
            if(it != _pEnt.end())
            {
                return it->second->addMatrix(matrix);
            }
            return {};
        }

        MOX_NO_DISCARD ErrorDataOutput<void> changeStatusE(const std::string& name , const bool status) noexcept
        {
            std::lock_guard<std::shared_mutex> lock(mutex);
            auto it = _pEnt.find(name);
            if(it != _pEnt.end())
            {
                return it->second->changeLightStatus(status);
            }
            return {};
        }

        const Model3* atM(const std::string& name) const noexcept
        {
            auto it = _pMd3.find(name);
            if(it != _pMd3.end())
            {
                return it->second.get();
            }            
            return nullptr;
        }

        Model3* atM(const std::string& name) noexcept
        {
            auto it = _pMd3.find(name);
            if(it != _pMd3.end())
            {
                return it->second.get();
            }            
            return nullptr;
        }

        const Entity* atE(const std::string& name) const noexcept
        {
            auto it = _pEnt.find(name);
            if(it != _pEnt.end())
            {
                return it->second.get();
            }            
            return nullptr;
        }

        Entity* atE(const std::string& name) noexcept
        {
            auto it = _pEnt.find(name);
            if(it != _pEnt.end())
            {
                return it->second.get();
            }            
            return nullptr;
        }

        Source::pVertexHeap heap{nullptr};

        ~EntityForge() noexcept
        {
            processDelete();
        }


    private:

        void processCopy(EntityForge&& input) noexcept
        {
            bluePrints = std::move(input.bluePrints);
            _pEnt = std::move(input._pEnt);
            _pMd3 = std::move(input._pMd3);
            heap = std::move(input.heap);
            inputData = input.inputData;
        }

        void processDelete() noexcept
        {
            _pEnt.clear();
            _pMd3.clear();
            heap = nullptr;
            inputData = {};
            Source::VertexHeap::Destroy();
        }
        std::vector<ForgeBluePrint> bluePrints{};
        CreateInfo inputData{};
        std::unordered_map<std::string , pEntity> _pEnt;
        std::unordered_map<std::string , pModel3> _pMd3;
        std::shared_mutex mutex{};

        MOX_NO_DISCARD ErrorDataOutput<void> _implExecuteBluePrints() noexcept
        {
            std::lock_guard<std::shared_mutex> lock(mutex);
            try
            {
                {
                    VertexHeapStart info{};
                    info.context = inputData.context;
                    info.txHub = inputData.txHub;
                    info.ltHub = inputData.lHub;
                    const auto result = Source::VertexHeap::Begin(info);
                    MOX_CHECK_RESULT(result)
                }

                for(auto& b : bluePrints)
                {
                    if(b.type == moxBluePrintEntity)
                    {
                        if(_pEnt.contains(b.name))
                        {
                            engineLogger(std::format("the Entity Forge already contains the blue print named -> {}" , b.name));
                            continue;
                        }

                        // one broken Entity does not stop the others : the error is in the log ( the constructor wrote it ) , and the forge goes on
                        try
                        {
                            const pEntity e = std::make_shared<Entity>(b.eInfo);
                            _pEnt.insert({b.name , e});
                        }
                        catch(const std::exception& e)
                        {
                            engineLogger(std::format("failed to create the Entity named -> {} , it is skipped , reason -> {}" , b.name , e.what()));
                        }
                        catch(...)
                        {
                            engineLogger(std::format("failed to create the Entity named -> {} , it is skipped , unknown exception" , b.name));
                        }
                    }
                    else if(b.type == moxBluePrintModel3)
                    {
                        if(_pMd3.contains(b.name))
                        {
                            engineLogger(std::format("the Entity Forge already contains the blue print named -> {}" , b.name));
                            continue;
                        }

                        // the same for the models : a broken model is skipped , the others are created
                        try
                        {
                            const pModel3 m = std::make_shared<Model3>(b.mInfo);
                            _pMd3.insert({b.name , m});
                        }
                        catch(const std::exception& e)
                        {
                            engineLogger(std::format("failed to create the Model3 named -> {} , it is skipped , reason -> {}" , b.name , e.what()));
                        }
                        catch(...)
                        {
                            engineLogger(std::format("failed to create the Model3 named -> {} , it is skipped , unknown exception" , b.name));
                        }
                    }
                    else MOX_RETURN_ERROR(MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to execute the blue print , since the type is undefined , Name: {}" , b.name));
                }
            }
            catch(std::exception &e)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_FAILED_CREATION , "failed to execute Blue prints");
            }

            {
                const auto result = Source::VertexHeap::End();
                MOX_CHECK_RESULT(result);
            }

            heap = std::make_shared<Source::VertexHeap>();
            bluePrints.clear();
            return {};
        }
    };

    MOX_DECLARE_PTR_STRUCTURE(EntityForge)
}
}