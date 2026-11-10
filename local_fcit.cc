#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>
#include <unordered_set>

#include <vamp/robots/panda.hh>

#include "fcit.hh"

namespace
{
    using Panda = vamp::robots::Panda;
    using Settings = vamp::planning::RoadmapSettings<vamp::planning::FCITStarNeighborParams>;
    // VAMP registers Sampler::Ptr itself as the Python RNG type.
    using Sampler = vamp::rng::RNG<Panda>;
    constexpr auto rake = vamp::FloatVectorWidth;
    using Planner = vamp::planning::FCIT<Panda, rake, Panda::resolution>;

    auto solve(
        const Panda::ConfigurationArray &start,
        const std::vector<Panda::ConfigurationArray> &goals,
        const vamp::collision::Environment<float> &environment,
        const Settings &settings,
        Sampler::Ptr sampler) -> vamp::planning::PlanningResult<Panda>
    {
        std::vector<Panda::Configuration> configurations(goals.begin(), goals.end());
        return Planner::solve(
            Panda::Configuration(start),
            configurations,
            vamp::collision::Environment<vamp::FloatVector<rake>>(environment),
            settings,
            sampler);
    }
}

NB_MODULE(local_fcit, module)
{
    nanobind::module_::import_("vamp");
    module.def("solve", &solve);
}
