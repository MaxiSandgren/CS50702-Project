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

    auto solve_anytime(
        const Panda::ConfigurationArray &start,
        const std::vector<Panda::ConfigurationArray> &goals,
        const vamp::collision::Environment<float> &environment,
        const Settings &settings,
        Sampler::Ptr sampler,
        double timeout_seconds,
        nanobind::callable callback,
        nanobind::object stop_check) -> vamp::planning::PlanningResult<Panda>
    {
        if (goals.size() != 1)
        {
            throw std::invalid_argument("Anytime mode requires exactly one goal.");
        }
        std::vector<Panda::Configuration> configurations(goals.begin(), goals.end());
        return Planner::solve(
            Panda::Configuration(start), configurations,
            vamp::collision::Environment<vamp::FloatVector<rake>>(environment),
            settings, sampler, timeout_seconds,
            [&callback](double seconds, float cost) { callback(seconds, cost); },
            [&stop_check]() { return not stop_check.is_none() and nanobind::cast<bool>(stop_check()); });
    }
}

NB_MODULE(local_fcit, module)
{
    nanobind::module_::import_("vamp");
    module.def("solve", &solve);
    module.def("solve_anytime", &solve_anytime,
               nanobind::arg("start"), nanobind::arg("goals"), nanobind::arg("environment"),
               nanobind::arg("settings"), nanobind::arg("sampler"), nanobind::arg("timeout_seconds"),
               nanobind::arg("callback"), nanobind::arg("stop_check") = nanobind::none());
}
