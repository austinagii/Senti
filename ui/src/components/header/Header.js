import Logo from "../logo/Logo";

import "./header.css";

export default function Header({ name }) {
  return (
    <>
      <section className="Header">
        <Logo className="Header-Logo"/>
        <p className="Header-UserText">Hey {name}</p>
        {/* <p>#1</p>
        <p>#2</p>
        <p>#3</p> */}
      </section>
    </>
  );
}
