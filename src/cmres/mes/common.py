from pandapipes.properties.fluids import get_fluid


def conversion_factor_kgps_to_mw(net):
    fcv = get_fluid(net).get_property("hhv")
    return fcv * 3600 / 1e3
